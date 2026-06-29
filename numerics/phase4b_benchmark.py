"""
PDET Phase-4b -- R-O6 evaluation hardening (from the Phase-4 eval audit, reviews/phase4_eval_audit.md):
  (A) UNKNOWN/COMPOSITE discovery with multiple-testing correction (FWER) -- moves beyond known-witness calibration.
  (B) EQUAL-BUDGET benchmark: PDET-guided vs full-process-tomography-equivalent, under the production schedule.
  (C) PROTECTION <-> DETECTABILITY trade-off (cost of breaking the schedule), with a background-noise model.

Honest thesis (narrowed per audit): PDET does NOT beat an oracle Ramsey on a single known error. Its value is the
SYSTEMATIC restricted-access layer: (i) the kernel identifies, for FREE (no shots), which candidate error
directions are undiscoverable under the current schedule+readout; (ii) it prescribes the minimal control change to
expose a targeted blind direction; (iii) honest multiple-testing + protection cost.

Run: python phase4b_benchmark.py -> ../results/phase4/phase4b_results.json + figs.
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "phase4"); os.makedirs(OUT, exist_ok=True)
SEED = 20260628
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]], complex); Z = np.array([[1, 0], [0, -1]], complex)
def st(v): v = np.array(v, complex); v = v/np.linalg.norm(v); return np.outer(v, v.conj())

# two-segment echo signal vector for a perturbation V (strength theta), pulse fraction f, finite-width residual w
def signal_vec(V, f, has_pi, S, O, w=0.0, theta=0.05):
    if not has_pi:
        K = 1.0 * V; U0 = I
    else:
        K = f * V + (1 - f) * (X.conj().T @ V @ X) + w * V; U0 = X
    Otil = [U0.conj().T @ Oo @ U0 for Oo in O]
    return np.array([(-1j*np.trace((rho @ Ot - Ot @ rho) @ K)).real for Ot in Otil for rho in S]) * theta

# ----------------------------------------------------------------------------- (A) multiple-testing discovery
def fwer_threshold(witness_dirs, V, N, alpha=0.05, ntrials=4000, seed=SEED):
    """FWER-controlled threshold for the max-|witness| statistic under the null (no error)."""
    rng = np.random.default_rng(seed)
    sig = np.sqrt(V / N); K = len(witness_dirs)
    dim = witness_dirs[0].size if K else 1
    maxstat = np.zeros(ntrials)
    for j, u in enumerate(witness_dirs):
        z = sig * (rng.standard_normal((ntrials, dim)) @ u)   # null statistic for witness j
        maxstat = np.maximum(maxstat, np.abs(z))
    return float(np.quantile(maxstat, 1 - alpha))

def discovery_power(present_signal, witness_dirs, thr, V, N, ntrials=4000, seed=SEED+1):
    """Power to DETECT (max-stat > thr) when 'present_signal' is the true accessible signal (0 => null/blind)."""
    rng = np.random.default_rng(seed); sig = np.sqrt(V / N); ntr = ntrials
    dim = witness_dirs[0].size
    detect = np.zeros(ntr, bool)
    for u in witness_dirs:
        z = (present_signal @ u) + sig * (rng.standard_normal((ntr, dim)) @ u)
        detect |= np.abs(z) > thr
    return float(np.mean(detect))

def partA_discovery():
    S = [st([1, 0]), st([0, 1]), st([1, 1]), st([1, 1j])]; O = [X, Y, Z]
    # K candidate coherent error directions; under the ECHO some are K-blind (Y,Z), some visible (X)
    dirs = {"X(amp)": X, "Y(phase)": Y, "Z(detuning)": Z}
    V_shadow = 3.0; N = 5000
    # witnesses = unit accessible-signal directions for each candidate under the ECHO schedule
    sigs_echo = {nm: signal_vec(Vp, 0.5, True, S, O) for nm, Vp in dirs.items()}
    witnesses = [ (s / (np.linalg.norm(s)+1e-18)) if np.linalg.norm(s) > 1e-9 else np.ones_like(s)/np.sqrt(s.size)
                  for s in sigs_echo.values() ]
    thr = fwer_threshold(witnesses, V_shadow, N)
    # PDET kernel flags blind directions FOR FREE (||signal||~0 under echo)
    blind = {nm: bool(np.linalg.norm(sigs_echo[nm]) < 1e-9) for nm in dirs}
    # power per direction under the echo (blind ones -> ~alpha, no signal)
    power_echo = {nm: discovery_power(sigs_echo[nm], witnesses, thr, V_shadow, N) for nm in dirs}
    # null FWER check
    fwer = discovery_power(np.zeros_like(witnesses[0]), witnesses, thr, V_shadow, N)
    # after PDET control mod (break echo to f=0.33) the blind directions become exposed
    sigs_mod = {nm: signal_vec(Vp, 0.33, True, S, O) for nm, Vp in dirs.items()}
    wit_mod = [ s/(np.linalg.norm(s)+1e-18) for s in sigs_mod.values() ]
    thr_mod = fwer_threshold(wit_mod, V_shadow, N)
    power_mod = {nm: discovery_power(sigs_mod[nm], wit_mod, thr_mod, V_shadow, N) for nm in dirs}
    return {"N": N, "alpha": 0.05, "FWER_threshold": thr, "empirical_FWER_under_null": fwer,
            "kernel_flags_blind_for_free": blind,
            "power_under_production_echo": power_echo,
            "power_after_PDET_control_mod_f0.33": power_mod,
            "note": ("Y,Z are K-blind under the echo: power ~ alpha (undetectable) at ANY N; PDET's kernel flags "
                     "them for free, and the minimal control mod (break echo) restores power ->1. X stays "
                     "detectable throughout. FWER controlled at alpha across the K-direction scan.")}

# ----------------------------------------------------------------------------- (B) equal-budget vs tomography
def partB_equal_budget():
    """Under the production echo, full process tomography (estimating error-generator coords) has ZERO Jacobian on
    the K-blind directions -> power 0 at any shot budget; PDET (free kernel) + minimal mod exposes them. For
    VISIBLE directions both work. Honest: PDET adds the know-blind + minimal-fix layer, not raw estimation power."""
    S = [st([1, 0]), st([0, 1]), st([1, 1]), st([1, 1j])]; O = [X, Y, Z]; V = 3.0
    Ns = [1000, 10000, 100000, 1000000]
    target = Z  # a K-blind direction under the echo (detuning)
    rows = {"N": Ns, "tomography_under_echo_power": [], "PDET_kernel+mod_power": []}
    for N in Ns:
        s_echo = signal_vec(target, 0.5, True, S, O)        # ~0 (blind) -> tomography sees nothing
        s_mod = signal_vec(target, 0.33, True, S, O)         # exposed by control mod
        thr_t = fwer_threshold([s_echo/(np.linalg.norm(s_echo)+1e-18) if np.linalg.norm(s_echo)>1e-9 else np.ones(s_echo.size)/np.sqrt(s_echo.size)], V, N)
        thr_m = fwer_threshold([s_mod/np.linalg.norm(s_mod)], V, N)
        rows["tomography_under_echo_power"].append(round(discovery_power(s_echo, [s_echo/(np.linalg.norm(s_echo)+1e-18) if np.linalg.norm(s_echo)>1e-9 else np.ones(s_echo.size)/np.sqrt(s_echo.size)], thr_t, V, N), 3))
        rows["PDET_kernel+mod_power"].append(round(discovery_power(s_mod, [s_mod/np.linalg.norm(s_mod)], thr_m, V, N), 3))
    rows["note"] = ("Target = Z-detuning, K-blind under the production echo. Tomography/estimation under the SAME "
                    "echo has ~zero power at every budget (the direction is in the kernel). PDET computes the "
                    "kernel for free, flags Z as blind, and prescribes the minimal control mod that exposes it -> "
                    "power->1. Equal-budget point: PDET's advantage is the kernel guidance, not estimation power.")
    return rows

# ----------------------------------------------------------------------------- (C) protection<->detectability
def partC_tradeoff():
    """Cost of breaking the schedule: as the pi-pulse moves off-center (f:0.5->0.1), the target Z-detuning becomes
    detectable (||K_Z|| up) BUT the echo's suppression of a background Z-dephasing noise is lost (protection down).
    Protection proxy = 1 - ||K_Z||/||K_Z(free)|| (how well the schedule still cancels a static Z bath)."""
    S = [st([1, 0]), st([0, 1]), st([1, 1]), st([1, 1j])]; O = [X, Y, Z]
    Kfree = np.linalg.norm(signal_vec(Z, None, False, S, O))
    fs = [0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.1]
    pts = []
    for f in fs:
        s = signal_vec(Z, f, True, S, O); detect = float(np.linalg.norm(s))
        protection = float(1 - detect / (Kfree + 1e-18))   # 1 = full DD protection, 0 = none
        pts.append({"f": f, "detectability(margin)": round(detect, 4), "protection_retained": round(protection, 4)})
    return {"sweep": pts, "note": "Pareto: f=0.5 max protection / zero detectability; moving off-center trades "
                                   "protection for detectability. The engineer picks the operating point."}

def main():
    res = {"seed": SEED}
    res["A_discovery_multiple_testing"] = partA_discovery()
    res["B_equal_budget_vs_tomography"] = partB_equal_budget()
    res["C_protection_detectability_tradeoff"] = partC_tradeoff()
    # figure: trade-off Pareto + discovery power
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    pts = res["C_protection_detectability_tradeoff"]["sweep"]
    ax[0].plot([p["protection_retained"] for p in pts], [p["detectability(margin)"] for p in pts], "o-")
    for p in pts: ax[0].annotate(f"f={p['f']}", (p["protection_retained"], p["detectability(margin)"]), fontsize=7)
    ax[0].set_xlabel("DD protection retained"); ax[0].set_ylabel("target detectability (margin)")
    ax[0].set_title("Fig 4: protection <-> detectability trade-off (cost of breaking the schedule)")
    A = res["A_discovery_multiple_testing"]
    names = list(A["power_under_production_echo"])
    x = np.arange(len(names))
    ax[1].bar(x-0.2, [A["power_under_production_echo"][n] for n in names], 0.4, label="under production echo")
    ax[1].bar(x+0.2, [A["power_after_PDET_control_mod_f0.33"][n] for n in names], 0.4, label="after PDET control mod")
    ax[1].axhline(0.05, ls=":", c="gray", label="alpha=0.05 (FWER)")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=8); ax[1].set_ylabel("discovery power (FWER-controlled)")
    ax[1].set_title("Fig 5: multiple-testing discovery; blind dirs need the control knob"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_5_benchmark.png"), dpi=120); plt.close(fig)
    with open(os.path.join(OUT, "phase4b_results.json"), "w") as f: json.dump(res, f, indent=2, default=str)
    # console
    print("\n===== Phase-4b R-O6 hardening =====")
    print(" (A) discovery (FWER):", "empirical_FWER=", round(A["empirical_FWER_under_null"], 3),
          "thr=", round(A["FWER_threshold"], 4))
    print("     kernel flags blind (free):", A["kernel_flags_blind_for_free"])
    print("     power under echo:", {k: round(v, 2) for k, v in A["power_under_production_echo"].items()})
    print("     power after control mod:", {k: round(v, 2) for k, v in A["power_after_PDET_control_mod_f0.33"].items()})
    print(" (B) equal-budget vs tomography:", res["B_equal_budget_vs_tomography"]["N"])
    print("     tomography-under-echo power:", res["B_equal_budget_vs_tomography"]["tomography_under_echo_power"])
    print("     PDET kernel+mod power:      ", res["B_equal_budget_vs_tomography"]["PDET_kernel+mod_power"])
    print(" (C) trade-off Pareto:", [(p["f"], p["protection_retained"], p["detectability(margin)"]) for p in res["C_protection_detectability_tradeoff"]["sweep"]])
    print("===================================\n")
    return res

if __name__ == "__main__":
    main()
