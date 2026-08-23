"""
PDET Phase-0 realistic device/control models (frozen params in phase0_spec.md).

Provides:
  - single transmon (qutrit) DRAG X90 schedule  (leakage level included)
  - two-qubit cross-resonance ZX(pi/2) schedule  (qubit subspace + classical crosstalk)
  - control-schedule augmentations: free / spin-echo / CPMG-2 (the A4 control knob)
  - the perturbation dictionary {V_j(t)} as per-step operator lists
  - restricted access model (S, O)

Units: angular freq in rad/ns; f = w/2pi quoted in GHz/MHz. ħ=1. dt in ns.
All schedules return (step_hams, dt) so they plug into pdet_core.Schedule and the exact a1 builder.
"""
from __future__ import annotations
import numpy as np
from pdet_core import qutrit_ops, qubit_ops, transmon_ops, ketbra, dag

TWO_PI = 2 * np.pi

# ----------------------------------------------------------------------------- frozen device params (spec §1)
ALPHA = TWO_PI * (-0.330)      # anharmonicity -330 MHz, rad/ns
GAMMA_REF = TWO_PI * 0.001     # 1 MHz reference perturbation rate, rad/ns
DRAG_BETA = 1.0
T_X90 = 35.0                   # ns
NSTEPS_1Q = 140                # 0.25 ns/step
T_CR = 300.0                   # ns
NSTEPS_2Q = 300                # 1 ns/step
J_CR = TWO_PI * 0.003          # 3 MHz

# Refocusing pulses inserted by the control knob have a FIXED PHYSICAL WIDTH in ns. They occupy steps of the
# schedule rather than being appended to it, so every augmentation of a given gate has the same total duration
# T, and refining the integration grid refines only the quadrature: the number of steps a pulse occupies grows
# with the grid so that its physical width is held fixed. Grid refinement is therefore a numerical operation
# and not a physical one.
T_PI_1Q = 2.0                  # ns, ideal pi-X(01) refocusing pulse inside the 35 ns X90
T_PI_2Q = 20.0                 # ns, pi-X on the control inside the 300 ns echoed cross-resonance

# ----------------------------------------------------------------------------- admissibility guard (G6)
# The control knob is only allowed to change the schedule, not the gate. Every builder below checks its own
# output against the gate it is named for before returning it, so no PDET number can be produced from a
# schedule that silently realizes a different unitary. This is the guard that catches an uncalibrated
# augmentation, which is exactly the failure the knob is most exposed to.
FID_TOL = 1e-3

def gate_fidelity(step_hams, dt, target, sub=None):
    """|Tr(target^dag U0(T))|/dim, restricted to the leading `sub` levels when the model carries a leakage level."""
    from pdet_core import Schedule
    U = Schedule(step_hams, dt).U0()
    if sub is not None:
        U = U[:sub, :sub]
    return abs(np.trace(dag(target) @ U)) / target.shape[0]

def _assert_admissible(step_hams, dt, target, sub, label, tol=FID_TOL):
    F = gate_fidelity(step_hams, dt, target, sub)
    if 1.0 - F > tol:
        raise AssertionError("schedule '%s' does not realize its target: infidelity %.3e > %.1e"
                             % (label, 1.0 - F, tol))
    return F

def _x90_target():
    c, s_ = np.cos(np.pi / 4), np.sin(np.pi / 4)
    return np.array([[c, -1j * s_], [-1j * s_, c]], dtype=complex)

def _zx90_target():
    I, X, Y, Z = qubit_ops()
    lam, Q = np.linalg.eigh(np.kron(Z, X))
    return Q @ np.diag(np.exp(-1j * (np.pi / 4) * lam)) @ dag(Q)

# =============================================================================== single transmon (qutrit)
def _gauss_env(t, T, sigma, area_target):
    g = np.exp(-((t - T / 2) ** 2) / (2 * sigma ** 2))
    return g

def _pulse_plan(nsteps, dt, n_pulses, t_pi):
    """
    Lay out n_pulses refocusing pulses of fixed physical width t_pi inside a schedule of nsteps steps.

    The pulse occupies m = round(t_pi/dt) steps, so halving dt doubles m and leaves the physical width alone.
    Pulses replace drive steps instead of extending the schedule, so the total duration is T for every
    augmentation. The drive is divided into n_pulses equal blocks and one pulse closes each block, which puts
    the last pulse flush against the end of the schedule and makes the inserted rotations pair up.

    Returns (mask, m) with mask[k] True where step k carries the pulse.
    """
    mask = np.zeros(nsteps, dtype=bool)
    if n_pulses == 0:
        return mask, 0
    m = max(1, int(round(t_pi / dt)))
    ndrive = nsteps - n_pulses * m
    if ndrive < n_pulses:
        raise ValueError("grid too coarse for %d pulses of width %g ns" % (n_pulses, t_pi))
    for k in range(1, n_pulses + 1):
        start = int(round(ndrive * k / n_pulses)) + (k - 1) * m
        mask[start:start + m] = True
    return mask, m

# a pi pulse count per augmentation. The counts are EVEN so the inserted rotations compose to the identity on
# the computational subspace and the augmented schedule realizes the same target as the unaugmented one, which
# is the admissibility constraint of Eq. (7).
NPULSE = {"free": 0, "echo": 2, "cpmg2": 4}

def transmon_x90_drag(nsteps=NSTEPS_1Q, T=T_X90, augment="free", t_pi=T_PI_1Q, nlev=3):
    """
    Rotating-frame qutrit DRAG X90. augment in {'free','echo','cpmg2'} inserts ideal pi-X(01) refocusing pulses
    of fixed width t_pi (the control knob), in even numbers and with the last one flush against the end of the
    schedule, so the net gate is the same X90 the unaugmented schedule realizes. The drive is off while a pulse
    is played and the envelope is recalibrated over the remaining drive time, so the rotation angle stays pi/2.
    Returns (step_hams, dt, meta).
    """
    I3, a, ad, n = transmon_ops(nlev)
    dt = T / nsteps
    mask, m = _pulse_plan(nsteps, dt, NPULSE[augment], t_pi)
    drive_idx = np.flatnonzero(~mask)
    nd = drive_idx.size
    Td = nd * dt

    # envelope laid out on the DRIVE timeline, so chopping out the pulses does not distort its shape
    tau = (np.arange(nd) + 0.5) * dt
    sigma = Td / 4.0
    raw = _gauss_env(tau, Td, sigma, None)
    # (a+ad)/2 has 0-1 matrix element 1/2, so the 01 rotation angle is the integral of Omega
    A = (np.pi / 2) / (np.sum(raw) * dt)
    Omega = A * raw
    OmegaDot = np.gradient(Omega, dt)

    Hdrift = (ALPHA / 2.0) * (ad @ ad @ a @ a)   # = alpha * |2><2|
    # ideal pi-X(01): generator restricted to the computational subspace, so the pulse itself adds no leakage
    X01 = ketbra(0, 1, nlev) + ketbra(1, 0, nlev)
    Hpi = (np.pi / (2.0 * m * dt)) * X01 if m else None

    step_hams = []
    j = 0
    for k in range(nsteps):
        if mask[k]:
            step_hams.append(Hdrift + Hpi)       # drift keeps acting while the pulse is played
        else:
            HI = Omega[j] * (a + ad) / 2.0
            HQ = -DRAG_BETA * OmegaDot[j] / ALPHA * 1j * (ad - a) / 2.0
            step_hams.append(Hdrift + HI + HQ)
            j += 1
    meta = {"augment": augment, "T": T, "dt": dt, "d": nlev, "kind": "transmon1q",
            "n_pulses": NPULSE[augment], "t_pi": t_pi if NPULSE[augment] else 0.0,
            "pulse_steps": int(mask.sum()), "pulses": np.flatnonzero(mask).tolist()}
    meta["gate_fidelity"] = _assert_admissible(step_hams, dt, _x90_target(), 2, "x90/" + augment)
    return step_hams, dt, meta

# =============================================================================== two-qubit cross-resonance
def cr_zx90(nsteps=NSTEPS_2Q, T=T_CR, augment="free", crosstalk=True, t_pi=T_PI_2Q, cancel_ix=True):
    """
    Cross-resonance ZX(pi/2). Qubit subspace (4-dim), with optional classical crosstalk (a spurious IX rate
    riding on the same drive). The Hamiltonian is H = (1/2) g_eff (Z(x)X) + (1/2) g_ct (I(x)X), matching
    Eq. (26), so the net rotation over the drive time is exp(-i (pi/4) ZX), the ZX90 the gate is named for.

    augment in {'free','echo','cpmg2'} builds the ECHOED variant: pi-X pulses of fixed width t_pi on the CONTROL
    qubit, each closing a drive block, together with a reversal of the drive phase. Conjugating by X(x)I sends
    ZX -> -ZX, so reversing the drive makes the ZX blocks add to the same net rotation while I(x)X, which the
    conjugation leaves alone, alternates in sign and refocuses. That is what the echo buys and it is why a pulse
    on the TARGET, which commutes with ZX, would refocus nothing.
    Returns (step_hams, dt, meta).
    """
    I, X, Y, Z = qubit_ops()
    def kron(A, B): return np.kron(A, B)
    ZX = kron(Z, X)
    IX = kron(I, X)
    XI = kron(X, I)
    dt = T / nsteps
    npul = NPULSE[augment]
    mask, m = _pulse_plan(nsteps, dt, npul, t_pi)
    nd = int((~mask).sum())
    Td = nd * dt
    # calibrate over the DRIVE time so the net ZX angle is pi/2 for every augmentation
    g_eff = (np.pi / 2) / Td
    # Classical crosstalk at ~15% of the CR rate. A calibrated gate cancels it with an active tone on the
    # target, which is what `cancel_ix` models, and which is what puts the unechoed and echoed schedules on the
    # SAME target so that comparing them is a comparison of schedules and not of gates, as Eq. (7) requires.
    # The crosstalk error DIRECTION stays in the dictionary; what is cancelled is the nominal term.
    g_ct_nom = (0.15 * g_eff) if crosstalk else 0.0
    g_ct = 0.0 if cancel_ix else g_ct_nom
    Hpi = (np.pi / (2.0 * m * dt)) * XI if m else None

    step_hams = []
    sign = 1.0
    for k in range(nsteps):
        if mask[k]:
            step_hams.append(Hpi)                  # exact pi rotation on the control, drive off
            if (k + 1 == nsteps) or (not mask[k + 1]):
                sign = -sign                       # reverse the drive phase once per pulse
        else:
            step_hams.append(sign * (0.5 * g_eff * ZX + 0.5 * g_ct * IX))
    meta = {"augment": augment, "T": T, "dt": dt, "d": 4, "kind": "cr2q", "crosstalk": crosstalk,
            "cancel_ix": cancel_ix, "g_ct_nominal": g_ct_nom, "g_ct_realized": g_ct,
            "n_pulses": npul, "t_pi": t_pi if npul else 0.0,
            "pulse_steps": int(mask.sum()), "pulses": np.flatnonzero(mask).tolist()}
    # With the crosstalk tone cancelled, every variant realizes ZX90 to the same tolerance, so the knob
    # comparison of sec:eval-2q is a comparison at fixed target. An uncancelled schedule is a deliberately
    # non-admissible ablation and is held only to a widened tolerance.
    meta["gate_fidelity"] = _assert_admissible(step_hams, dt, _zx90_target(), None, "cr/" + augment,
                                               tol=FID_TOL if (npul or cancel_ix) else 1e-2)
    return step_hams, dt, meta

# =============================================================================== perturbation dictionary
def transmon_idle(nsteps=NSTEPS_1Q, T=T_X90):
    """Idle schedule: anharmonic drift only, NO drive. A probe where many directions are Z-readout-invisible."""
    I3, a, ad, n = qutrit_ops()
    dt = T / nsteps
    Hdrift = (ALPHA / 2.0) * (ad @ ad @ a @ a)
    step_hams = [Hdrift.copy() for _ in range(nsteps)]
    meta = {"augment": "idle", "T": T, "dt": dt, "pulses": [], "d": 3, "kind": "transmon1q_idle"}
    return step_hams, dt, meta

def dictionary_1q(nsteps_eff, sched_step_hams=None, nlev=3):
    """Per-step V_j lists, CONSTANT operators (schedule-independent) for clean cross-schedule compare."""
    I3, a, ad, n = transmon_ops(nlev)
    NS = nsteps_eff
    Vs = {
        "amp":   [GAMMA_REF * (a + ad) / 2.0 for _ in range(NS)],          # in-phase drive amplitude error
        "det":   [GAMMA_REF * n for _ in range(NS)],                       # detuning / frequency error
        "phase": [GAMMA_REF * 1j * (ad - a) / 2.0 for _ in range(NS)],     # quadrature / phase error
        "leak":  [GAMMA_REF * (ketbra(1, 2, nlev) + ketbra(2, 1, nlev)) for _ in range(NS)],  # leakage 1<->2
        "wd":    [GAMMA_REF * (a + ad) / 2.0 * (1.0 if k < NS // 2 else 0.6) for k in range(NS)],  # waveform dist.
    }
    return Vs

def dictionary_2q(nsteps_eff):
    I, X, Y, Z = qubit_ops()
    def kron(A, B): return np.kron(A, B)
    NS = nsteps_eff
    Vs = {
        "amp_c":  [GAMMA_REF * kron(X, I) / 1.0 for _ in range(NS)],   # control amplitude
        "det_c":  [GAMMA_REF * kron(Z, I) for _ in range(NS)],
        "ctk":    [GAMMA_REF * kron(I, X) for _ in range(NS)],          # classical crosstalk on target
        "spec":   [GAMMA_REF * kron(Z, Z) for _ in range(NS)],          # spectator ZZ
        "det_t":  [GAMMA_REF * kron(I, Z) for _ in range(NS)],          # target detuning
    }
    return Vs

# =============================================================================== access models (spec §3)
def access_1q(level="Z"):
    """
    level='Z'   : computational readout only  O={Z}   (realistic MINIMAL engineering access)
    level='ZX'  : O={Z,X}                                (moderately restricted; spec default)
    level='rich': O={Z,X,Y,leak-pop}                     (rich contrast; should shrink ker M -> falsifier v)
    """
    I3, a, ad, n = qutrit_ops()
    def q2(psi2):
        v = np.zeros(3, dtype=complex); v[:2] = psi2; return np.outer(v, v.conj())
    S = [q2([1, 0]), q2([0, 1]), q2([1, 1] / np.sqrt(2)), q2([1, 1j] / np.sqrt(2))]
    Z3 = np.diag([1.0, -1.0, 0.0]).astype(complex)
    X3 = (a + ad); X3 = X3 / np.linalg.norm(X3, 2)
    Y3 = 1j * (ad - a); Y3 = Y3 / np.linalg.norm(Y3, 2)
    leakpop = np.diag([0.0, 0.0, 1.0]).astype(complex)
    O = {"Z": [Z3], "ZX": [Z3, X3], "rich": [Z3, X3, Y3, leakpop]}[level]
    return S, O

def access_2q(rich=False, computational=False):
    I, X, Y, Z = qubit_ops()
    def kron(A, B): return np.kron(A, B)
    def st(v): v = np.array(v, dtype=complex); v = v / np.linalg.norm(v); return np.outer(v, v.conj())
    ket = {"0": [1, 0], "1": [0, 1], "+": [1, 1]}
    S = []
    for c in ["0", "1", "+"]:
        for t in ["0", "1", "+"]:
            S.append(st(np.kron(ket[c], ket[t])))
    if computational:
        O = [kron(Z, I), kron(I, Z), kron(Z, Z)]                            # computational readout
    elif not rich:
        O = [kron(Z, I), kron(I, Z), kron(Z, Z), kron(X, I), kron(I, X)]    # + control-qubit transverse
    else:
        O = [kron(P, Q) for P in [I, X, Y, Z] for Q in [I, X, Y, Z]][1:]  # all nontrivial 2-local Paulis
    return S, O
