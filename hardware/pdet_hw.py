"""PDET on hardware: schedule construction, circuit builders, and analysis.

Everything here is built at the SCHEDULE level -- delays, X/Y pulses and virtual
Z frame changes. No pulse-level control is used or needed, which matters because
Qiskit Pulse was removed in Qiskit 2.0 and IBM backends no longer expose it.

The physics being reproduced is the paper's idle-qubit use case: an XY4 sequence
protecting a memory qubit makes a static Z-drift first-order invisible to full
single-qubit tomography (a control-level blind spot), and a small asymmetry in
the sequence exposes it at a predicted finite shot cost.

Two things make the hardware version faithful rather than merely suggestive:

1. The drift is INJECTED as a known quantity via virtual-Z frame updates, so the
   ground truth is exact. Because the injected drift is along sigma_z and so is
   the dominant dephasing generator, the two commute and lumping a segment's
   continuous rotation into discrete frame updates leaves the density matrix
   unchanged -- this is an identity, not an approximation. Only amplitude
   damping fails to commute, which is why segments are subdivided.

2. Every duration is solved on the backend's own timing grid before any circuit
   is built. A pi-pulse that lands off centre because a delay got rounded is
   indistinguishable from the asymmetry we deliberately introduce, so an
   unsatisfiable grid is raised as an error rather than silently rounded away.

Convention note: the injected drift accumulates during the FREE segments only,
not during the pi-pulses. This matches `numerics/selfcheck_dd_idle_usecase.py`,
where the pulse steps propagate under the drive Hamiltonian alone; keeping the
conventions identical is what makes the simulated and measured numbers
comparable.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit

# --------------------------------------------------------------------------
# Schedules. Fractions are exact so the grid solver can reason about them.
# XY4 places its pi-pulses at 1,3,5,7 x T/8; the asymmetric variant moves the
# first one to 0.8 x T/8 = T/10, which is the "control edit" the workflow
# prescribes.
# --------------------------------------------------------------------------
XY4_FRACS = (Fraction(1, 8), Fraction(3, 8), Fraction(5, 8), Fraction(7, 8))
XY4_AXES = ("x", "y", "x", "y")

SCHEDULES: Dict[str, Tuple[Tuple[Fraction, ...], Tuple[str, ...]]] = {
    "free":      ((), ()),
    "echo":      ((Fraction(1, 2),), ("x",)),
    "xy4":       (XY4_FRACS, XY4_AXES),
    "xy4_drop1": (XY4_FRACS[:3], XY4_AXES[:3]),
    "xy4_asym":  ((Fraction(1, 10),) + XY4_FRACS[1:], XY4_AXES),
}

BASES = ("X", "Y", "Z")
# The setting the restricted-access kernel selects as the witness for a Z-detuning on an idle
# window. Estimators use this rather than the empirical argmax over BASES: the argmax of three
# noisy separations is biased away from zero, so on a schedule whose response is not resolved it
# returns a finite, acquisition-size-dependent cost under a true null.
WITNESS = "Y"


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Timing:
    """The backend timing facts every duration has to satisfy."""
    dt: float           # seconds per dt
    granularity: int    # every pulse/delay length must be a multiple of this (dt)
    x_dur: int          # duration of an X pulse (dt)

    def to_dt(self, seconds: float) -> int:
        return int(round(seconds / self.dt))

    def to_seconds(self, n_dt: int) -> float:
        return n_dt * self.dt

    def to_us(self, n_dt: int) -> float:
        return n_dt * self.dt * 1e6


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def window_quantum(fracs: Iterable[Fraction], granularity: int) -> int:
    """Smallest window length in dt for which every pulse centre lands on the grid.

    A centre sits at ``frac * T``. Writing ``frac = a/b`` in lowest terms, we need
    ``T * a / b`` to be a multiple of ``granularity``, i.e.
    ``T = 0 (mod granularity*b / gcd(a, granularity*b))``. The window must satisfy
    this for every fraction at once, so take the lcm.
    """
    q = 1
    for f in fracs:
        a, b = f.numerator, f.denominator
        need = granularity * b // gcd(a, granularity * b)
        q = _lcm(q, need)
    return q


def solve_window(timing: Timing, target_seconds: float,
                 schedules: Sequence[str] = tuple(SCHEDULES)) -> int:
    """Pick the window length (dt) nearest `target_seconds` that every schedule can use.

    Solved once for the union of all schedules so that every variant runs in an
    identical window -- comparing protection or N* across schedules of different
    lengths would confound the schedule with the exposure time.
    """
    fracs = [f for name in schedules for f in SCHEDULES[name][0]]
    if not fracs:
        fracs = [Fraction(1, 1)]
    q = window_quantum(fracs, timing.granularity)
    target_dt = timing.to_dt(target_seconds)
    n = max(1, int(round(target_dt / q)))
    return n * q


def split_delay(total_dt: int, n_parts: int, granularity: int) -> List[int]:
    """Split a delay into `n_parts` grid-legal pieces summing exactly to `total_dt`.

    Sub-dividing lets the injected phase be applied incrementally, which bounds
    the error from the one channel that does not commute with a Z-drift
    (amplitude damping). The pieces are as equal as the grid allows.
    """
    if total_dt <= 0:
        return [0] * n_parts
    if total_dt % granularity:
        raise ValueError("delay %d dt is not a multiple of granularity %d"
                         % (total_dt, granularity))
    units = total_dt // granularity
    if units < n_parts:                      # too short to subdivide meaningfully
        return [total_dt] + [0] * (n_parts - 1)
    base, extra = divmod(units, n_parts)
    return [(base + (1 if i < extra else 0)) * granularity for i in range(n_parts)]


@dataclass(frozen=True)
class Layout:
    """A schedule laid out on the grid: alternating free segments and pi-pulses."""
    name: str
    window_dt: int
    segments: Tuple[int, ...]     # free-evolution durations (dt), len = n_pulses + 1
    axes: Tuple[str, ...]         # pi-pulse axis per gap
    centres_dt: Tuple[int, ...]   # intended pulse centres (dt), for verification

    @property
    def free_dt(self) -> int:
        return sum(self.segments)


def layout_schedule(name: str, timing: Timing, window_dt: int) -> Layout:
    """Place the pi-pulses so their CENTRES sit at the schedule's fractions.

    The pulse has real width, so half of it is taken out of the delay before the
    centre and half out of the delay after. Skipping this shifts every pulse late
    by x_dur/2 -- a systematic that looks exactly like the asymmetry the
    experiment is trying to measure.
    """
    fracs, axes = SCHEDULES[name]
    half = timing.x_dur // 2
    if timing.x_dur % 2 or half % timing.granularity:
        raise ValueError(
            "X duration %d dt does not split evenly onto a %d-dt grid; "
            "pulse centres cannot be placed exactly"
            % (timing.x_dur, timing.granularity))

    centres = []
    for f in fracs:
        c = f * window_dt
        if c.denominator != 1:
            raise ValueError("centre %s is not an integer number of dt" % c)
        centres.append(int(c))

    segments = []
    cursor = 0
    for c in centres:
        d = (c - half) - cursor
        if d < 0:
            raise ValueError("pulses overlap: centre %d dt is too close to %d dt" % (c, cursor))
        if d % timing.granularity:
            raise ValueError("free segment %d dt is off the %d-dt grid" % (d, timing.granularity))
        segments.append(d)
        cursor = c + half
    tail = window_dt - cursor
    if tail < 0:
        raise ValueError("last pulse runs past the end of the window")
    if tail % timing.granularity:
        raise ValueError("tail segment %d dt is off the %d-dt grid" % (tail, timing.granularity))
    segments.append(tail)

    return Layout(name=name, window_dt=window_dt, segments=tuple(segments),
                  axes=tuple(axes), centres_dt=tuple(centres))


# --------------------------------------------------------------------------
# Circuit construction
# --------------------------------------------------------------------------
def _pi_pulse(qc: QuantumCircuit, q: int, axis: str) -> None:
    """A pi rotation about x or y. Y is X conjugated by free virtual-Z frames."""
    if axis == "x":
        qc.x(q)
    elif axis == "y":
        qc.rz(-np.pi / 2, q)
        qc.x(q)
        qc.rz(np.pi / 2, q)
    else:
        raise ValueError("axis must be 'x' or 'y', got %r" % axis)


def _prep_plus(qc: QuantumCircuit, q: int) -> None:
    """|0> -> |+>. Written with h so the intent is legible; the transpiler emits sx/rz."""
    qc.h(q)


def _measure_basis(qc: QuantumCircuit, q: int, basis: str) -> None:
    """Rotate `basis` onto Z before the computational-basis readout."""
    if basis == "X":
        qc.h(q)
    elif basis == "Y":
        qc.sdg(q)
        qc.h(q)
    elif basis != "Z":
        raise ValueError("basis must be X, Y or Z, got %r" % basis)


def build_circuit(layout: Layout, timing: Timing, qubits: Sequence[int],
                  theta_rad_per_us: float, basis: str, n_qubits_total: int,
                  n_sub: int = 8) -> QuantumCircuit:
    """One DD circuit: prepare |+>, run the schedule with an injected Z-drift, read out.

    `theta_rad_per_us` is the drift's angular rate in the same units the
    simulation uses (rad/us, so theta=0.05 is 5e4 rad/s ~ 7.96 kHz). The phase is
    accumulated over the free segments only, matching the simulation's
    convention, and is applied as virtual-Z frame updates, which are exact and
    cost no time.

    The same schedule runs on every qubit in `qubits` inside one circuit: these
    are independent idle-memory experiments, so the chip parallelises them for
    free and the per-qubit statistics of H4 cost no extra QPU time.
    """
    qc = QuantumCircuit(n_qubits_total, len(qubits))
    # theta is an angular rate in rad/us, so keep the arithmetic in microseconds:
    # a free interval of `piece` dt accumulates theta * (piece * us_per_dt) radians
    # of relative phase, which is exactly rz(phase).
    us_per_dt = timing.dt * 1e6

    for q in qubits:
        _prep_plus(qc, q)

    for i, seg in enumerate(layout.segments):
        pieces = split_delay(seg, n_sub, timing.granularity)
        for piece in pieces:
            if piece == 0:
                continue
            phase = theta_rad_per_us * piece * us_per_dt
            for q in qubits:
                qc.delay(piece, q, unit="dt")
                if phase:
                    qc.rz(phase, q)
        if i < len(layout.axes):
            for q in qubits:
                _pi_pulse(qc, q, layout.axes[i])

    for q in qubits:
        _measure_basis(qc, q, basis)
    for c, q in enumerate(qubits):
        qc.measure(q, c)
    return qc


def verify_layout(layout: Layout, timing: Timing) -> Dict[str, float]:
    """Check the built schedule really puts the pulses where the schedule says.

    Returns the realised centre of each pulse in dt and its error against the
    intended centre. Any non-zero error means a timing bug, and a timing bug in
    this experiment forges the signal.
    """
    half = timing.x_dur // 2
    cursor = 0
    realised = []
    for i, seg in enumerate(layout.segments):
        cursor += seg
        if i < len(layout.axes):
            realised.append(cursor + half)
            cursor += timing.x_dur
    errs = [r - c for r, c in zip(realised, layout.centres_dt)]
    return {
        "realised_centres_dt": realised,
        "intended_centres_dt": list(layout.centres_dt),
        "centre_errors_dt": errs,
        "max_abs_error_dt": max([abs(e) for e in errs], default=0),
        "total_dt": cursor,
        "window_dt": layout.window_dt,
        "length_error_dt": cursor - layout.window_dt,
    }


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------
def expectation_from_counts(counts: Dict[str, int], bit: int,
                            p_readout: float = 0.0) -> Tuple[float, int]:
    """<P> for one qubit from a counts dict, corrected for symmetric readout error.

    Returns (expectation, shots). The correction divides by (1 - 2p), which also
    inflates the variance by 1/(1-2p)^2 -- the same factor the paper carries.
    """
    shots = 0
    signed = 0
    for bitstring, n in counts.items():
        clean = bitstring.replace(" ", "")
        # Qiskit orders classical bits little-endian in the printed string
        val = clean[::-1][bit]
        signed += n * (1 if val == "0" else -1)
        shots += n
    if shots == 0:
        return 0.0, 0
    e = signed / shots
    if p_readout:
        e = e / (1.0 - 2.0 * p_readout)
    return e, shots


def signal_slope(thetas: Sequence[float], signals: Sequence[Sequence[float]],
                 weights: Sequence[float] | None = None) -> Dict[str, float]:
    """Fit d<s>/dtheta at theta=0 across a SIGNED theta sweep.

    The sweep is signed on purpose: fitting a slope through zero separates the
    linear response from any theta-independent offset (SPAM, static ZZ), which a
    one-sided sweep cannot do.

    `signals` is one row per theta, one column per observable. Returns the
    per-observable slopes, the norm of the slope vector, and its standard error.
    """
    t = np.asarray(thetas, dtype=float)
    s = np.asarray(signals, dtype=float)
    if s.ndim == 1:
        s = s[:, None]
    w = np.ones_like(t) if weights is None else np.asarray(weights, dtype=float)

    # weighted least squares of s = a + b*t, per column
    W = w.sum()
    tbar = (w * t).sum() / W
    dt_ = t - tbar
    denom = (w * dt_ ** 2).sum()
    slopes, ses = [], []
    for j in range(s.shape[1]):
        y = s[:, j]
        ybar = (w * y).sum() / W
        b = (w * dt_ * (y - ybar)).sum() / denom
        resid = y - (ybar + b * dt_)
        dof = max(len(t) - 2, 1)
        sigma2 = (w * resid ** 2).sum() / dof
        slopes.append(b)
        ses.append(np.sqrt(sigma2 / denom))
    slopes = np.array(slopes)
    ses = np.array(ses)
    norm = float(np.linalg.norm(slopes))
    # error propagation for the norm of the slope vector
    norm_se = float(np.sqrt(((slopes / norm) ** 2 * ses ** 2).sum())) if norm > 0 else float(ses.max())
    return {"slopes": slopes.tolist(), "slope_ses": ses.tolist(),
            "norm": norm, "norm_se": norm_se}


def two_point_nstar(mean_h0: float, mean_h1: float, var_per_shot: float,
                    alpha: float = 0.05) -> float:
    """The paper's two-point cost N* = (2 z_alpha)^2 V / gamma^2 for one witness setting."""
    from scipy.stats import norm as _norm
    gamma = abs(mean_h1 - mean_h0)
    if gamma == 0:
        return float("inf")
    z = _norm.ppf(1 - alpha)
    return float((2 * z) ** 2 * var_per_shot / gamma ** 2)


def bootstrap_detection_rates(shots_h0: np.ndarray, shots_h1: np.ndarray,
                              n_grid: Sequence[int], n_trials: int = 2000,
                              seed: int = 20260628) -> List[Dict[str, float]]:
    """Empirical (false-alarm, miss) vs shot count, by resampling two acquired pools.

    `shots_h0`/`shots_h1` are per-shot +/-1 outcomes on the witness setting. For
    each N we draw sub-batches of size N from each pool and run the two-point
    test with the threshold at the midpoint of the two pool means.

    Resampling one pool captures shot noise exactly. It does NOT capture
    run-to-run drift, which is why the pools are re-acquired on separate days and
    the two numbers reported separately.
    """
    rng = np.random.default_rng(seed)
    m0, m1 = shots_h0.mean(), shots_h1.mean()
    thr = 0.5 * (m0 + m1)
    sign = 1.0 if m1 > m0 else -1.0

    # Drawing N shots with replacement from a pool of +/-1 values and averaging is
    # exactly a Binomial: with a fraction p of +1 in the pool, the sub-batch has
    # k ~ Binomial(N, p) of them and mean (2k - N)/N. Sampling k directly is the
    # same distribution as materialising the draws, and costs O(n_trials) instead
    # of O(n_trials * N) -- at N = 3e5 the naive form allocates 6e8 indices per
    # grid point and never finishes.
    p0 = float((shots_h0 > 0).mean())
    p1 = float((shots_h1 > 0).mean())

    out = []
    for N in n_grid:
        N = int(N)
        s0 = (2.0 * rng.binomial(N, p0, size=n_trials) - N) / N
        s1 = (2.0 * rng.binomial(N, p1, size=n_trials) - N) / N
        fa = float(np.mean(sign * (s0 - thr) > 0))
        miss = float(np.mean(sign * (s1 - thr) <= 0))
        out.append({"N": N, "false_alarm": fa, "miss": miss})
    return out


def nstar_from_rates(rates: Sequence[Dict[str, float]], alpha: float = 0.05,
                     beta: float = 0.05) -> float:
    """Smallest N on the grid where both empirical error rates clear their targets."""
    for r in rates:
        if r["false_alarm"] <= alpha and r["miss"] <= beta:
            return float(r["N"])
    return float("inf")


# --------------------------------------------------------------------------
# Backend introspection
# --------------------------------------------------------------------------
def timing_from_backend(backend, qubit: int = 0) -> Timing:
    """Read dt, the alignment grid and the X duration off a real or fake backend."""
    target = backend.target
    dt = backend.dt if backend.dt is not None else target.dt
    granularity = getattr(target, "granularity", 1) or 1
    pulse_alignment = getattr(target, "pulse_alignment", 1) or 1
    grid = _lcm(int(granularity), int(pulse_alignment))

    x_seconds = None
    for gate in ("x", "sx"):
        try:
            props = target[gate][(qubit,)]
        except (KeyError, TypeError):
            continue
        if props is not None and props.duration:
            x_seconds = props.duration if gate == "x" else 2 * props.duration
            break
    if x_seconds is None:
        raise RuntimeError("backend reports no X/SX duration for qubit %d" % qubit)

    x_dur = int(round(x_seconds / dt))
    # round the pulse up onto the grid, and keep its half on the grid too so the
    # centres stay placeable
    step = 2 * grid
    if x_dur % step:
        x_dur += step - (x_dur % step)
    return Timing(dt=dt, granularity=grid, x_dur=x_dur)


def pick_qubits(backend, n: int, min_distance: int = 2) -> List[int]:
    """Choose `n` qubits that are pairwise at least `min_distance` apart on the coupling map.

    Spacing them keeps ZZ crosstalk between the qubits under test negligible, so
    each one is a genuinely independent replica; spectators we do not use stay in
    |0> and contribute a static shift that experiment H5 measures on purpose.
    Within that constraint, prefer qubits with the longest T2.
    """
    target = backend.target
    n_qubits = backend.num_qubits

    neighbours = {q: set() for q in range(n_qubits)}
    cmap = target.build_coupling_map()
    if cmap is not None:
        for a, b in cmap.get_edges():
            neighbours[a].add(b)
            neighbours[b].add(a)

    def t2_of(q):
        try:
            props = backend.properties()
            return props.t2(q) or 0.0
        except Exception:
            return 0.0

    order = sorted(range(n_qubits), key=lambda q: -t2_of(q))

    def within(q, chosen):
        # breadth-first to `min_distance - 1` hops
        frontier, seen = {q}, {q}
        for _ in range(min_distance - 1):
            nxt = set()
            for x in frontier:
                nxt |= neighbours[x]
            frontier = nxt - seen
            seen |= nxt
        return bool(seen & set(chosen))

    chosen: List[int] = []
    for q in order:
        if len(chosen) >= n:
            break
        if not within(q, chosen):
            chosen.append(q)
    return sorted(chosen)
