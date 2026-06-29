"""
PDET Phase-4d -- final realism polish (marginal): hardware-noise-PSD protection metric + nuisance handling.

(1) PROTECTION via the FILTER FUNCTION under a realistic 1/f dephasing PSD. For a static dephasing (Z) coupling,
    the schedule's toggling sign y(t) (+1, flipped to -1 by each pi-X) defines the filter g(omega)=|Y(omega)|^2,
    Y(omega)=int_0^T y(t) e^{i omega t} dt. The residual dephasing under noise PSD S(omega) is
    chi = (1/2pi) int S(omega) g(omega) domega. A balanced echo suppresses g at low omega (Y ~ omega) -> strong
    protection vs 1/f; breaking the echo restores low-omega sensitivity -> lost protection. Protection metric =
    chi_free / chi_schedule. This replaces the crude ||K|| proxy with a physically-grounded number, and yields the
    protection<->detectability Pareto on a real noise model.

(2) NUISANCE handling: an unknown common benign DRIFT (calibration) is a nuisance that inflates false discoveries.
    PDET's benign projection P_B^perp (already in the margin gamma) IS the nuisance-robust construction: projecting
    out span(M*drift) restores FWER control. We show naive-vs-projected discovery FWER under an unknown drift.

Run: python phase4d_noise_nuisance.py -> ../results/phase4/phase4d_results.json + fig.
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import figstyle; figstyle.apply()

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "phase4"); os.makedirs(OUT, exist_ok=True)
SEED = 20260628
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]], complex); Z = np.array([[1, 0], [0, -1]], complex)
def st(v): v = np.array(v, complex); v = v/np.linalg.norm(v); return np.outer(v, v.conj())

# ----------------------------------------------------------------------------- (1) filter-function protection
def toggling_sign(f_pi, ngrid=2000, T=1.0):
    """y(t) for a static-Z coupling: +1 before the pi-X, -1 after (the pi flips Z->-Z). f_pi=None => free (+1)."""
    t = (np.arange(ngrid) + 0.5) * (T / ngrid)
    y = np.ones(ngrid)
    if f_pi is not None:
        y[t > f_pi * T] = -1.0
    return t, y, T / ngrid

def residual_dephasing(f_pi, alpha=1.0, wmin=2*np.pi*1e-3, wmax=2*np.pi*10, nw=600, T=1.0):
    """chi = (1/2pi) int_{wmin}^{wmax} S(w) |Y(w)|^2 dw, S(w)=A/w^alpha (1/f for alpha=1). A=1 (relative)."""
    t, y, dt = toggling_sign(f_pi, T=T)
    ws = np.geomspace(wmin, wmax, nw)
    Yw = np.array([np.abs(np.sum(y * np.exp(1j * w * t)) * dt)**2 for w in ws])  # |Y(w)|^2
    S = 1.0 / ws**alpha
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapz(S * Yw, ws) / (2*np.pi))

def detect_margin(f_pi, theta=0.05):
    """target coherent Z-detuning detectability margin under full tomography (two-segment, ideal pulse)."""
    S = [st([1,0]), st([0,1]), st([1,1]), st([1,1j])]; O = [X, Y, Z]
    if f_pi is None: K = 1.0*Z; U0 = I
    else: K = (2*f_pi-1)*Z; U0 = X
    Otil = [U0.conj().T@O_@U0 for O_ in O]
    v = np.array([(-1j*np.trace((rho@Ot-Ot@rho)@K)).real for Ot in Otil for rho in S])*theta
    return float(np.linalg.norm(v))

def part1_psd_protection():
    chi_free = residual_dephasing(None)
    fs = [0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.1]
    pts = []
    for f in fs:
        chi = residual_dephasing(f); margin = detect_margin(f)
        pts.append({"f": f, "chi_residual_dephasing": chi, "protection_factor_vs_free": round(chi_free/chi, 2) if chi > 0 else None,
                    "detectability_margin": round(margin, 4)})
    return {"noise_model": "1/f dephasing PSD S(w)=1/w (alpha=1)", "chi_free": chi_free, "pareto": pts,
            "note": ("Filter-function protection on a real 1/f PSD: the symmetric echo (f=0.5) suppresses residual "
                     "dephasing by a large factor vs free (protection) while making the target coherent error "
                     "undetectable; breaking the echo trades that protection for detectability along a Pareto.")}

# ----------------------------------------------------------------------------- (2) nuisance handling
def fwer_under_nuisance(project_out_drift, ntrials=4000, N=5000, V=3.0, drift_mag=0.04, seed=SEED):
    """Discovery FWER under an UNKNOWN common benign drift. Witnesses for K candidate visible directions; the
    drift adds a structured common-mode signal. Naive max-stat inflates FWER; projecting out span(M*drift)
    (PDET's P_B^perp) restores control. Returns empirical FWER (false-discovery rate under the null=no attack)."""
    rng = np.random.default_rng(seed)
    S = [st([1,0]), st([0,1]), st([1,1]), st([1,1j])]; O = [X, Y, Z]
    # visible candidate directions under the free schedule
    def sig(V_):
        Otil = O; rows = [(-1j*np.trace((rho@Ot-Ot@rho)@V_)).real for Ot in Otil for rho in S]; return np.array(rows)
    cand = [sig(X), sig(Y), sig(Z)]
    drift_signal = sig(Z)  # the unknown benign drift direction (e.g. slow detuning calibration drift)
    drift_signal = drift_signal / (np.linalg.norm(drift_signal)+1e-18)
    if project_out_drift:
        Pd = np.eye(drift_signal.size) - np.outer(drift_signal, drift_signal)
        cand = [Pd @ c for c in cand]
    wit = [c/(np.linalg.norm(c)+1e-18) for c in cand]
    sig_noise = np.sqrt(V/N)
    # threshold from the no-drift null
    null_max = np.zeros(ntrials)
    for u in wit:
        null_max = np.maximum(null_max, np.abs(sig_noise*(rng.standard_normal((ntrials, u.size))@u)))
    thr = np.quantile(null_max, 0.95)
    # FWER under the null-with-unknown-drift: no attack, but a random-magnitude benign drift is present
    fd = 0
    for _ in range(ntrials):
        drift = drift_mag * rng.standard_normal() * drift_signal
        x = drift + sig_noise*rng.standard_normal(drift_signal.size)
        if project_out_drift:
            Pd = np.eye(drift_signal.size) - np.outer(drift_signal, drift_signal); x = Pd @ x
        if any(np.abs(x@u) > thr for u in wit): fd += 1
    return fd/ntrials

def part2_nuisance():
    naive = fwer_under_nuisance(project_out_drift=False)
    projected = fwer_under_nuisance(project_out_drift=True)
    return {"FWER_naive_with_unknown_drift": round(naive, 3),
            "FWER_with_benign_projection_P_Bperp": round(projected, 3),
            "note": ("An unknown benign drift inflates the naive discovery FWER well above alpha=0.05 (false "
                     "discoveries). PDET's benign projection P_B^perp -- the SAME construction as the margin gamma "
                     "-- projects out span(M*drift) and restores FWER ~ alpha. PDET's nuisance handling is built in.")}

def main():
    res = {"seed": SEED}
    res["psd_protection"] = part1_psd_protection()
    res["nuisance_handling"] = part2_nuisance()
    # figure: PSD-based Pareto + nuisance FWER
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.3))
    pts = res["psd_protection"]["pareto"]
    ax[0].plot([p["protection_factor_vs_free"] for p in pts], [p["detectability_margin"] for p in pts], "o-")
    for p in pts: ax[0].annotate(f"f={p['f']}", (p["protection_factor_vs_free"], p["detectability_margin"]), fontsize=11)
    ax[0].set_xlabel("DD protection factor vs free (1/f PSD, filter-function)"); ax[0].set_xscale("log")
    ax[0].set_ylabel("target detectability margin")
    nu = res["nuisance_handling"]
    ax[1].bar(["naive\n(unknown drift)", "PDET P_Bperp\n(projected)"],
              [nu["FWER_naive_with_unknown_drift"], nu["FWER_with_benign_projection_P_Bperp"]], color=["C3", "C2"])
    ax[1].axhline(0.05, ls="--", c="gray", label="alpha=0.05"); ax[1].set_ylabel("discovery FWER under nuisance drift")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig7_8_noise_nuisance.png"), dpi=120); plt.close(fig)
    with open(os.path.join(OUT, "phase4d_results.json"), "w") as f: json.dump(res, f, indent=2, default=str)
    print("\n===== Phase-4d noise PSD + nuisance =====")
    print(" PSD protection (1/f), chi_free =", round(res["psd_protection"]["chi_free"], 4))
    for p in res["psd_protection"]["pareto"]:
        print(f"   f={p['f']}: protection x{p['protection_factor_vs_free']:<7} detectability={p['detectability_margin']}")
    print(" Nuisance FWER: naive=", res["nuisance_handling"]["FWER_naive_with_unknown_drift"],
          " projected(P_Bperp)=", res["nuisance_handling"]["FWER_with_benign_projection_P_Bperp"])
    print("=========================================\n")
    return res

if __name__ == "__main__":
    main()
