"""Genuine paper content: (1) false-alarm/miss detection curves vs shots N (direct vs classical-shadow estimator);
(2) device-realism parameter sweep: detection cost N* of a visible error vs coherence time T2. Real Monte-Carlo.
Produces fig_detection_curves.png, fig_t2_sweep.png + JSON, copied into paper/figures."""
import os, json
import numpy as np
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "selfcheck"); os.makedirs(OUT, exist_ok=True)
PD = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
SEED = 20260628

def detection_curves(gamma=0.1, ntrials=8000, seed=SEED):
    """Benign(0) vs attack(margin gamma) on a 1-D witness; FA and miss vs N for direct (V=1) and shadow (V=3)."""
    rng = np.random.default_rng(seed)
    Ns = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    out = {"N": Ns}
    for label, V in [("direct (V=1)", 1.0), ("shadow (V=3)", 3.0)]:
        FA, MISS = [], []
        for N in Ns:
            sig = np.sqrt(V / N); thr = gamma / 2
            fa = np.mean(rng.normal(0, sig, ntrials) > thr)
            miss = np.mean(rng.normal(gamma, sig, ntrials) <= thr)
            FA.append(float(fa)); MISS.append(float(miss))
        out[label] = {"FA": FA, "MISS": MISS}
    out["gamma"] = gamma
    return out

def t2_sweep(gamma0=0.2, T_seq_us=16.0, V=3.0, alpha=0.05):
    """A visible error's first-order signal damps ~ exp(-T_seq/T2); detection cost N* = V(2z)^2 / (gamma0 e^{-T/T2})^2.
    Sweep T2 over the realistic transmon range; report N* (shots) -- shows how coherence sets the achievable budget."""
    z = norm.ppf(1 - alpha); T2s = [40, 60, 80, 120, 200, 400]
    rows = {"T2_us": T2s, "Nstar": []}
    for T2 in T2s:
        margin = gamma0 * np.exp(-T_seq_us / T2)
        rows["Nstar"].append(float(V * (2 * z) ** 2 / margin ** 2))
    return rows

def main():
    res = {"detection_curves": detection_curves(), "t2_sweep": t2_sweep()}
    dc = res["detection_curves"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 2.9))
    for label, mk in [("direct (V=1)", "o-"), ("shadow (V=3)", "s--")]:
        ax.loglog(dc["N"], np.array(dc[label]["MISS"]) + 1e-4, mk, label=f"miss, {label}")
    ax.loglog(dc["N"], np.array(dc["direct (V=1)"]["FA"]) + 1e-4, "^:", color="gray", label="false alarm (direct)")
    ax.axhline(0.05, ls=":", c="r", alpha=0.6, label="5% target")
    ax.set_xlabel("shots $N$"); ax.set_ylabel("error rate"); ax.set_title(f"Detection error vs shots ($\\gamma={dc['gamma']}$)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(PD, "fig_detection_curves.png"), dpi=130); plt.close(fig)
    ts = res["t2_sweep"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 2.9))
    ax.semilogy(ts["T2_us"], ts["Nstar"], "o-")
    ax.set_xlabel("coherence time $T_2$ ($\\mu$s)"); ax.set_ylabel("detection cost $N^\\star$ (shots)")
    ax.set_title("Device realism: detection cost vs coherence time"); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(PD, "fig_t2_sweep.png"), dpi=130); plt.close(fig)
    json.dump(res, open(os.path.join(OUT, "detection_t2.json"), "w"), indent=2)
    print("detection FA(direct):", [round(x,3) for x in dc["direct (V=1)"]["FA"]])
    print("detection MISS(direct):", [round(x,3) for x in dc["direct (V=1)"]["MISS"]])
    print("T2 sweep N*:", [round(x,1) for x in ts["Nstar"]])
if __name__ == "__main__": main()
