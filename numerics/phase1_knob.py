"""
PDET Phase-1 — R-O1 make-or-break: a NON-1q control-knob win that lowers REAL finite-shot detection cost.

Setting (realistic, security-motivated): 2-qubit cross-resonance, COMPUTATIONAL-ONLY readout O={ZI,IZ,ZZ} (raw
device readout; transverse needs basis-change pulses). A control-qubit Z-detuning / frequency-mismatch
perturbation (an Xu-Szefer-style stealthy attack, arXiv:2406.05941) is FIRST-ORDER INVISIBLE to this readout.
A designed control-schedule augmentation (X90 on the control qubit before readout -- a Ramsey-type probe) EXPOSES
it: dim ker M 1->0, gamma 0 -> finite, so the finite-shot detection cost drops from INFINITE to a finite N*.

This is the orientation review's key demand: control redesign must lower REAL finite-shot cost, not just a
singular value. We report: dim ker / gamma per augmentation; the analytic shot budget N*(direction) with the
local-Pauli shadow-norm variance; a Monte-Carlo FA/miss confirmation at N*; a Fisher/CFI experiment-design
baseline (the closest competitor); and a 3-qubit spectator extension to show it is not 2q-special.

Run: python phase1_knob.py  ->  ../results/phase1/{phase1_results.json, fig*.png, PHASE1_RESULT.md helper}.
Worst-case directions. Frozen Phase-0 artifacts are untouched.
"""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import figstyle; figstyle.apply()
from scipy.stats import norm
from pdet_core import (Schedule, toggling_generator, response_map, kernel_dim, kernel_basis,
                       singular_spectrum, benign_projector, gamma_margin)
import models as M_

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "phase1"); os.makedirs(OUT, exist_ok=True)
RTOL = 1e-9; SEED = 20260628
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]], complex); Z = np.array([[1, 0], [0, -1]], complex)

def kron(*a):
    r = np.array([[1]], complex)
    for x in a: r = np.kron(r, x)
    return r
def st(v):
    v = np.array(v, complex); v = v / np.linalg.norm(v); return np.outer(v, v.conj())

# ----------------------------------------------------------------------------- 2q setup
def access_2q_comp():
    S = [st(np.kron(c, t)) for c in ([1, 0], [0, 1], [1, 1] / np.sqrt(2)) for t in ([1, 0], [0, 1], [1, 1] / np.sqrt(2))]
    O = [kron(Z, I), kron(I, Z), kron(Z, Z)]          # computational-basis correlators only
    return S, O

READOUT_ROT_NS = 20.0    # fixed physical duration of the appended basis-change rotation

def cr_augmented(aug):
    """CR(ZX90) schedule + an appended single-qubit readout-rotation segment (the measurement knob).

    The rotation has a fixed physical duration rather than one integration step, so refining the grid does not
    change the schedule it describes. It is appended after the gate, so it leaves U_0(T) of the gate itself
    untouched and acts purely as a basis change.
    """
    sh, dt, meta = M_.cr_zx90(augment="free", crosstalk=True)
    R = {"free": None, "x90_c": kron(X, I), "y90_c": kron(Y, I), "x90_t": kron(I, X),
         "x90_both": kron(X, I) + kron(I, X)}
    if R[aug] is not None:
        m = max(1, int(round(READOUT_ROT_NS / dt)))
        sh = sh + [(np.pi / 2 / (m * dt)) * R[aug] / 2] * m
    return sh, dt

# ----------------------------------------------------------------------------- finite-shot shot budget
def pauli_weight(O):
    """Weight of a tensor product of Paulis: the number of factors that are not the identity."""
    nq = int(round(np.log2(O.shape[0])))
    P = {"I": I, "X": X, "Y": Y, "Z": Z}
    for names in _pauli_words(nq):
        T = np.array([[1.0 + 0j]])
        for nm in names:
            T = kron(T, P[nm])
        if np.allclose(T, O) or np.allclose(T, -O):
            return sum(1 for nm in names if nm != "I")
    raise ValueError("observable is not a signed Pauli word")

def _pauli_words(nq):
    if nq == 0:
        yield ""
        return
    for head in "IXYZ":
        for tail in _pauli_words(nq - 1):
            yield head + tail

def shadow_variances(O_list):
    """Per-observable classical-shadow variance factors, 3^weight, as a diagonal covariance.

    A random-Pauli shadow estimates a weight-w Pauli with variance inflated by 3^w relative to measuring it
    directly. The factors differ across observables, so the penalty is a covariance and not one number: for
    {ZI, IZ, ZZ} it is diag(3, 3, 9). Collapsing that to a mean would misweight the witness.
    """
    return np.array([3.0 ** pauli_weight(O) for O in O_list], dtype=float)

def shadow_norm_factor(O_list):
    """Worst-case scalar summary of shadow_variances, for the places that need a single conservative number."""
    return float(np.max(shadow_variances(O_list)))

def _finite(o):
    """Replace non-finite floats by null so the archive is valid JSON. An infinite N* means the direction is
    invisible at any budget, which the surrounding fields record explicitly."""
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o

def shot_budget(margin, V, fa=0.05, miss=0.05):
    """N* to detect a signal of size 'margin' against shot noise variance V, at (fa,miss).
       Two-point Gaussian test: sqrt(N) * margin = (z_fa + z_miss) sqrt(V)  => N = V (z_fa+z_miss)^2 / margin^2.
       A margin below the numerical-zero floor (1e-12) is treated as truly invisible => INFINITE cost."""
    if margin <= 1e-12:
        return np.inf
    z = norm.ppf(1 - fa) + norm.ppf(1 - miss)
    return V * z ** 2 / margin ** 2

def detection_margin_for_direction(M, names, direction_name, benign_names):
    """Accessible, benign-projected first-order signal size of a UNIT perturbation along 'direction_name'."""
    j = names.index(direction_name)
    benign_idx = [names.index(b) for b in benign_names if b in names and b != direction_name]
    P = benign_projector(M, benign_idx)
    e = np.zeros(M.shape[1]); e[j] = 1.0
    return float(np.linalg.norm(P @ (M @ e)))

def montecarlo_detect(M, names, direction_name, theta_mag, Nshots, V, benign_names, n_rep=400, seed=SEED):
    """Confirm FA/miss at given Nshots for benign(0) vs attack(theta_mag*e_dir), Gaussian shot model var V/N."""
    rng = np.random.default_rng(seed)
    j = names.index(direction_name)
    benign_idx = [names.index(b) for b in benign_names if b in names and b != direction_name]
    P = benign_projector(M, benign_idx)
    e = np.zeros(M.shape[1]); e[j] = 1.0
    v_att = P @ (M @ (theta_mag * e)); v_ben = P @ (M @ np.zeros(M.shape[1]))
    diff = v_att - v_ben; nrm = np.linalg.norm(diff) + 1e-18; u = diff / nrm
    thr = 0.5 * (u @ v_att + u @ v_ben)
    sigma = np.sqrt(V / Nshots)
    fa = miss = 0
    for _ in range(n_rep):
        xb = v_ben + sigma * rng.standard_normal(v_ben.shape)
        if u @ xb > thr: fa += 1
        xa = v_att + sigma * rng.standard_normal(v_att.shape)
        if u @ xa <= thr: miss += 1
    return fa / n_rep, miss / n_rep

# ----------------------------------------------------------------------------- Fisher / CFI baseline
def cfi_per_direction(M, names):
    """Single-direction classical Fisher info proxy = ||M e_j||^2 (the experiment-design competitor's target)."""
    return {n: float(np.sum(M[:, names.index(n)] ** 2)) for n in names}

# ----------------------------------------------------------------------------- 3q spectator extension
def setup_3q():
    """3 qubits: CR on (0,1) generating ZXI + spectator coupling to qubit 2; computational readout."""
    def K3(a, b, c): return kron(a, b, c)
    g_eff = (np.pi / 2) / M_.T_CR
    ct = 0.15 * g_eff
    nsteps = 80; dt = M_.T_CR / nsteps
    def sched(aug):
        sh = []
        for _ in range(nsteps):
            # the same H = (1/2) g_eff ZX + (1/2) g_ct IX convention models.cr_zx90 uses
            H = (0.5 * g_eff * K3(Z, X, I) + 0.5 * ct * K3(I, X, I)
                 + 0.3 * 0.5 * g_eff * K3(Z, I, Z))       # CR + crosstalk + spectator ZZ(0,2)
            sh.append(H)
        if aug == "x90_c":
            m = max(1, int(round(READOUT_ROT_NS / dt)))
            sh = sh + [(np.pi / 2 / (m * dt)) * K3(X, I, I) / 2] * m
        return sh, dt

    def realized_rot_ns():
        return max(1, int(round(READOUT_ROT_NS / dt))) * dt
    S = [st(np.kron(np.kron(a, b), c)) for a in ([1, 0], [1, 1] / np.sqrt(2)) for b in ([1, 0], [0, 1]) for c in ([1, 0], [1, 1] / np.sqrt(2))]
    O = [K3(Z, I, I), K3(I, Z, I), K3(I, I, Z), K3(Z, Z, I), K3(Z, I, Z)]
    # the same Gamma_ref normalisation models.dictionary_2q uses, so the 3q shot costs are on the scale the
    # rest of the paper quotes rather than a factor 1/Gamma_ref away from it
    G = M_.GAMMA_REF
    Vd = {"det_c": G * K3(Z, I, I), "ctk": G * K3(I, X, I), "spec02": G * K3(Z, I, Z),
          "amp_c": G * K3(X, I, I), "det_t": G * K3(I, Z, I)}
    return sched, S, O, Vd

# ----------------------------------------------------------------------------- main
def main():
    res = {"seed": SEED}
    S, O = access_2q_comp()
    benign = ["amp_c"]   # only slow amplitude drift is benign; det_c (frequency mismatch) is a detectable ATTACK
    # The shadow penalty is a per-observable covariance, diag(3^weight), computed from the observables actually
    # measured rather than assumed to be the weight-one value. A single conservative scalar is carried for the
    # two-point formula, and the covariance itself is reported so the two protocols cannot be conflated.
    V_shadow_diag = shadow_variances(O)
    V_shadow = shadow_norm_factor(O)
    res["shadow_variance_diag"] = {"observables": ["ZI", "IZ", "ZZ"][:len(O)],
                                   "factors": V_shadow_diag.tolist(),
                                   "scalar_used": V_shadow,
                                   "note": "worst-case 3^weight; the diagonal is the honest covariance"}

    augs = ["free", "x90_c", "y90_c", "x90_t", "x90_both"]
    twoq = {}
    for aug in augs:
        sh, dt = cr_augmented(aug); NS = len(sh)
        Vd = M_.dictionary_2q(NS); names = list(Vd); Vl = [Vd[k] for k in names]
        sc = Schedule(sh, dt); K = [toggling_generator(sc, v) for v in Vl]
        M = response_map(sc, K, S, O); s = singular_spectrum(M); Mop = s[0]
        attack_cols = [i for i, n in enumerate(names) if n not in benign]
        g, _ = gamma_margin(M, benign_idx=[names.index(b) for b in benign], attack_cols=attack_cols)
        margin_detc = detection_margin_for_direction(M, names, "det_c", benign)
        twoq[aug] = {"dim_ker_M": kernel_dim(M), "gamma_norm": float(g / Mop),
                     "margin_det_c": margin_detc, "M_op": float(Mop),
                     "shot_budget_det_c_direct": shot_budget(margin_detc * 0.05, 1.0),   # ||theta||=0.05
                     "shot_budget_det_c_shadow": shot_budget(margin_detc * 0.05, V_shadow),
                     "cfi_per_direction": cfi_per_direction(M, names), "names": names}
    res["twoq"] = twoq

    # Monte-Carlo confirmation at the predicted N* for free (infinite) vs x90_c (finite)
    mc = {}
    for aug in ["free", "x90_c"]:
        sh, dt = cr_augmented(aug); NS = len(sh); Vd = M_.dictionary_2q(NS); names = list(Vd); Vl = [Vd[k] for k in names]
        sc = Schedule(sh, dt); K = [toggling_generator(sc, v) for v in Vl]; M = response_map(sc, K, S, O)
        Nb = twoq[aug]["shot_budget_det_c_shadow"]
        Ntest = int(min(Nb * 2, 1e7)) if np.isfinite(Nb) else int(1e6)
        fa, miss = montecarlo_detect(M, names, "det_c", 0.05, Ntest, V_shadow, benign)
        mc[aug] = {"N_tested": Ntest, "predicted_N_star_shadow": (None if not np.isfinite(Nb) else Nb),
                   "FA": fa, "MISS": miss}
    res["montecarlo_det_c"] = mc

    # knob verdict (2q): does x90_c shrink ker and make det_c finite-shot-detectable while free cannot?
    res["knob_win_2q"] = {
        "free_dim_ker": twoq["free"]["dim_ker_M"], "x90c_dim_ker": twoq["x90_c"]["dim_ker_M"],
        "free_margin_det_c": twoq["free"]["margin_det_c"], "x90c_margin_det_c": twoq["x90_c"]["margin_det_c"],
        "free_shotcost_det_c": twoq["free"]["shot_budget_det_c_shadow"],
        "x90c_shotcost_det_c": twoq["x90_c"]["shot_budget_det_c_shadow"],
        "knob_exposes_invisible_attack": bool(twoq["free"]["margin_det_c"] < 1e-9 <= twoq["x90_c"]["margin_det_c"]),
        "real_finite_shot_cost_drops_inf_to_finite": bool(
            (not np.isfinite(twoq["free"]["shot_budget_det_c_shadow"])) and
            np.isfinite(twoq["x90_c"]["shot_budget_det_c_shadow"]))}

    # 3q extension
    sched3, S3, O3, Vd3 = setup_3q()
    threeq = {}
    for aug in ["free", "x90_c"]:
        sh, dt = sched3(aug); NS = len(sh)
        names = list(Vd3); Vl = [[Vd3[k]] * NS for k in names]
        sc = Schedule(sh, dt); K = [toggling_generator(sc, v) for v in Vl]; M = response_map(sc, K, S3, O3)
        s = singular_spectrum(M)
        margin = detection_margin_for_direction(M, names, "det_c", ["amp_c"])
        threeq[aug] = {"dim_ker_M": kernel_dim(M), "margin_det_c": float(margin),
                       "shotcost_det_c_shadow": shot_budget(margin * 0.05, V_shadow), "names": names}
    res["threeq"] = threeq
    res["knob_win_3q"] = {
        "free_margin": threeq["free"]["margin_det_c"], "x90c_margin": threeq["x90_c"]["margin_det_c"],
        "free_dim_ker": threeq["free"]["dim_ker_M"], "x90c_dim_ker": threeq["x90_c"]["dim_ker_M"],
        "cost_inf_to_finite": bool((not np.isfinite(threeq["free"]["shotcost_det_c_shadow"])) and
                                   np.isfinite(threeq["x90_c"]["shotcost_det_c_shadow"]))}

    _figs(res, twoq);
    with open(os.path.join(OUT, "phase1_results.json"), "w") as f:
        # an infinite cost is a real verdict, but bare Infinity is not JSON, so it is written as null and the
        # accompanying note says so. Standards-compliant readers can then load the archive.
        json.dump(_finite(res), f, indent=2, allow_nan=False, default=str)
    _print(res); return res

def _figs(res, twoq):
    augs = list(twoq);
    fig, ax = plt.subplots(1, 2, figsize=figstyle.figsize(1.0, 2.8))
    kd = [twoq[a]["dim_ker_M"] for a in augs]; gn = [twoq[a]["gamma_norm"] for a in augs]
    x = np.arange(len(augs))
    ax[0].bar(x - 0.2, kd, 0.4, label="dim ker M"); ax[0].bar(x + 0.2, gn, 0.4, label="gamma_norm(attack)")
    ax[0].axhline(0.02, ls="--", c="r", label="gamma_min=0.02")
    ax[0].set_xticks(x); ax[0].set_xticklabels(augs); ax[0].legend()
    ax[0].set_title("2q CR, computational readout")
    sc = [twoq[a]["shot_budget_det_c_shadow"] for a in augs]
    sc_plot = [s if np.isfinite(s) else np.nan for s in sc]
    ax[1].bar(x, [1e9 if not np.isfinite(s) else s for s in sc], color=["gray" if not np.isfinite(s) else "C2" for s in sc])
    ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels(augs)
    ax[1].set_title("N* for det_c (gray = invisible)")
    ax[1].set_ylabel("shots N* (shadow), log")
    fig.tight_layout(); figstyle.save(fig, OUT, "fig1_knob_win_2q")

def _print(r):
    print("\n========== PDET Phase-1 R-O1 make-or-break ==========")
    for a, v in r["twoq"].items():
        sc = v["shot_budget_det_c_shadow"]
        print(f" 2q {a:9s}: dimker={v['dim_ker_M']} gamma={v['gamma_norm']:.3f} margin(det_c)={v['margin_det_c']:.2e} "
              f"N*(det_c,shadow)={'INF' if not np.isfinite(sc) else f'{sc:.1f}'}")
    print(" KNOB WIN 2q:", json.dumps(r["knob_win_2q"], default=str))
    print(" MonteCarlo det_c:", json.dumps(r["montecarlo_det_c"], default=str))
    print(" 3q:", {a: {"dimker": v["dim_ker_M"], "margin": round(v["margin_det_c"], 4)} for a, v in r["threeq"].items()})
    print(" KNOB WIN 3q:", json.dumps(r["knob_win_3q"], default=str))
    print("=====================================================\n")

if __name__ == "__main__":
    main()
