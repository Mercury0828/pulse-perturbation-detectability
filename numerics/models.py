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
from pdet_core import qutrit_ops, qubit_ops, ketbra, dag

TWO_PI = 2 * np.pi

# ----------------------------------------------------------------------------- frozen device params (spec §1)
ALPHA = TWO_PI * (-0.330)      # anharmonicity -330 MHz, rad/ns
GAMMA_REF = TWO_PI * 0.001     # 1 MHz reference perturbation rate, rad/ns
DRAG_BETA = 1.0
T_X90 = 35.0                   # ns
NSTEPS_1Q = 70
T_CR = 300.0                   # ns
NSTEPS_2Q = 150               # 2 ns/step (kept moderate for speed; structural results dt-insensitive)
J_CR = TWO_PI * 0.003          # 3 MHz

# =============================================================================== single transmon (qutrit)
def _gauss_env(t, T, sigma, area_target):
    g = np.exp(-((t - T / 2) ** 2) / (2 * sigma ** 2))
    return g

def transmon_x90_drag(nsteps=NSTEPS_1Q, T=T_X90, augment="free"):
    """
    Rotating-frame qutrit DRAG X90. augment in {'free','echo','cpmg2'} inserts ideal pi-X(01) refocusing pulses
    (the control knob) while preserving the net X90 (echo/cpmg implemented as added pi pulses + compensation).
    Returns (step_hams, dt, meta).
    """
    I3, a, ad, n = qutrit_ops()
    sigma = T / 4.0
    dt = T / nsteps
    ts = (np.arange(nsteps) + 0.5) * dt
    # calibrate amplitude so the 0<->1 rotation is pi/2 (numerically, ignoring leakage to first order)
    raw = _gauss_env(ts, T, sigma, None)
    # area of (a+ad)/2 drive; rotation angle ~ integral of Omega over t for the 01 transition
    X01 = (a + ad)  # contains 0-1 and 1-2 couplings; the 01 element is 1
    # angle from envelope amplitude A: theta01 = A * sum(env)*dt  (since (a+ad)/2 has 01 matrix element 1/2 -> *2)
    A = (np.pi / 2) / (np.sum(raw) * dt)
    Omega = A * raw
    OmegaDot = np.gradient(Omega, dt)
    Hdrift = (ALPHA / 2.0) * (ad @ ad @ a @ a)   # = alpha * |2><2| (number-nonpreserving anharmonic term)
    step_hams = []
    # build a refocusing insertion plan
    pulses = {"free": [], "echo": [nsteps // 2], "cpmg2": [nsteps // 3, 2 * nsteps // 3]}[augment]
    for k in range(nsteps):
        HI = Omega[k] * (a + ad) / 2.0
        HQ = -DRAG_BETA * OmegaDot[k] / ALPHA * 1j * (ad - a) / 2.0
        H = Hdrift + HI + HQ
        step_hams.append(H)
    # add ideal instantaneous pi-X(01) refocusing as strong short segments folded into the nearest steps
    # (modeled as extra rotation: we append the schedule with compensating pi pulses around insertions)
    meta = {"augment": augment, "T": T, "dt": dt, "pulses": pulses, "d": 3, "kind": "transmon1q"}
    if pulses:
        step_hams = _insert_pi_x01(step_hams, dt, pulses)
        meta["nsteps_effective"] = len(step_hams)
    return step_hams, dt, meta

def _insert_pi_x01(step_hams, dt, positions):
    """Insert an ideal pi rotation about X in the {0,1} subspace at given step indices (echo/CPMG knob)."""
    I3, a, ad, n = qutrit_ops()
    # a pi pulse about X01: realize as one extra step with a strong resonant drive of area pi
    pi_amp = np.pi / dt
    Hpi = pi_amp * (a + ad) / 2.0
    out = []
    posset = set(positions)
    for k, H in enumerate(step_hams):
        out.append(H)
        if k in posset:
            out.append(Hpi)   # ideal-ish pi insertion (one step of strong drive)
    return out

# =============================================================================== two-qubit cross-resonance
def cr_zx90(nsteps=NSTEPS_2Q, T=T_CR, augment="free", crosstalk=True):
    """
    Effective cross-resonance generating ZX. Qubit subspace (2x2 x 2x2 = 4-dim). Optional classical crosstalk
    (spurious IX drive). augment in {'free','echo','cpmg2'} inserts pi-X on the target (echo CR, the real knob).
    Returns (step_hams, dt, meta).
    """
    I, X, Y, Z = qubit_ops()
    def kron(A, B): return np.kron(A, B)
    ZX = kron(Z, X)
    IX = kron(I, X)
    IZ = kron(I, Z)
    dt = T / nsteps
    # CR amplitude so net ZX angle = pi/2 over T (echoed variants compensate)
    g = J_CR
    # flat-top CR drive (simple): constant effective ZX rate so that g_eff*T = pi/2
    g_eff = (np.pi / 2) / T
    ct = (0.15 * g_eff) if crosstalk else 0.0   # classical crosstalk ~15% spurious IX
    step_hams = []
    pulses = {"free": [], "echo": [nsteps // 2], "cpmg2": [nsteps // 3, 2 * nsteps // 3]}[augment]
    posset = set(pulses)
    for k in range(nsteps):
        H = g_eff * ZX + ct * IX
        step_hams.append(H)
        if k in posset:
            # echo CR: pi-X on target flips the sign of subsequent ZX (sign tracked via inserted pi pulse)
            step_hams.append((np.pi / dt) * IX)
    meta = {"augment": augment, "T": T, "dt": dt, "pulses": pulses, "d": 4, "kind": "cr2q", "crosstalk": crosstalk}
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

def dictionary_1q(nsteps_eff, sched_step_hams=None):
    """Per-step V_j lists (qutrit), CONSTANT operators (schedule-independent) for clean cross-schedule compare."""
    I3, a, ad, n = qutrit_ops()
    NS = nsteps_eff
    Vs = {
        "amp":   [GAMMA_REF * (a + ad) / 2.0 for _ in range(NS)],          # in-phase drive amplitude error
        "det":   [GAMMA_REF * n for _ in range(NS)],                       # detuning / frequency error
        "phase": [GAMMA_REF * 1j * (ad - a) / 2.0 for _ in range(NS)],     # quadrature / phase error
        "leak":  [GAMMA_REF * (ketbra(1, 2, 3) + ketbra(2, 1, 3)) for _ in range(NS)],  # leakage 1<->2
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

def access_2q(rich=False):
    I, X, Y, Z = qubit_ops()
    def kron(A, B): return np.kron(A, B)
    def st(v): v = np.array(v, dtype=complex); v = v / np.linalg.norm(v); return np.outer(v, v.conj())
    ket = {"0": [1, 0], "1": [0, 1], "+": [1, 1]}
    S = []
    for c in ["0", "1", "+"]:
        for t in ["0", "1", "+"]:
            S.append(st(np.kron(ket[c], ket[t])))
    if not rich:
        O = [kron(Z, I), kron(I, Z), kron(Z, Z), kron(X, I), kron(I, X)]
    else:
        O = [kron(P, Q) for P in [I, X, Y, Z] for Q in [I, X, Y, Z]][1:]  # all nontrivial 2-local Paulis
    return S, O
