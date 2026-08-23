"""Build the manuscript's hardware figures from the raw CCQC counts.

Reads `results_hw/*.json` and writes vector PDFs straight into `paper/figures/`,
at the final printed size, using the same `figstyle` the simulation figures use.

    python make_paper_figures.py

Every number plotted here is recomputed from the raw counts, so the figures and
the text cannot drift apart.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "numerics"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle

import pdet_hw as P

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results_hw")
OUT = os.path.join(HERE, "..", "paper", "figures")
P_RO = 0.006
THETA_TEST = 0.05

# frozen before acquisition, from results/selfcheck/dd_idle_usecase_results.json
PRE_SLOPE = {"free": 14.00, "xy4_drop1": 3.33, "xy4_asym": 0.700, "xy4": 2e-5}
PRE_NSTAR = 9310.4
LABEL = {"free": "free", "xy4": "XY4", "xy4_drop1": "XY4 drop-1", "xy4_asym": "asym. XY4"}


def newest(pattern):
    hits = sorted(glob.glob(os.path.join(RES, pattern)))
    if not hits:
        raise SystemExit("no file matching %s" % pattern)
    return json.load(open(hits[-1]))


def h1_slopes(d):
    """Debiased response norm per schedule per qubit, plus the typical component error."""
    qs, ths = d["config"]["qubits"], d["config"]["theta_sweep"]
    sig = defaultdict(lambda: defaultdict(dict))
    for r in d["records"]:
        for bit, q in enumerate(qs):
            e, _ = P.expectation_from_counts(r["counts"], bit, P_RO)
            sig[r["schedule"]][q].setdefault(r["theta"], {})[r["basis"]] = e
    deb, raw, ses = defaultdict(dict), defaultdict(dict), []
    for s in sig:
        for q in qs:
            rows = [[sig[s][q][t][b] for b in P.BASES] for t in ths]
            f = P.signal_slope(ths, rows)
            v2 = float(np.mean(np.square(f["slope_ses"])))
            ses.append(np.sqrt(v2))
            raw[s][q] = f["norm"]
            # E[norm^2] = |true|^2 + 3 sigma^2 for a 3-component noisy vector
            deb[s][q] = float(np.sqrt(max(f["norm"] ** 2 - 3 * v2, 0.0)))
    return qs, deb, raw, float(np.median(ses))


WITNESS = "Y"      # the setting the restricted-access kernel selects for a Z-detuning on an idle window


def h2_nstar(d):
    qs = d["config"]["qubits"]
    mean = defaultdict(dict)
    for r in d["records"]:
        for bit, q in enumerate(qs):
            e, _ = P.expectation_from_counts(r["counts"], bit, P_RO)
            mean[(r["schedule"], r["basis"], r["theta"])][q] = e
    out, argmax_agrees = {}, {}
    for s in ("xy4_asym", "xy4"):
        vals, agree = [], 0
        for q in qs:
            diffs = {b: mean[(s, b, THETA_TEST)][q] - mean[(s, b, 0.0)][q] for b in P.BASES}
            # the workflow prescribes the witness; taking the empirical argmax instead biases the
            # separation away from zero and manufactures a finite cost on an unresolved schedule
            if max(diffs, key=lambda b: abs(diffs[b])) == WITNESS:
                agree += 1
            m0, m1 = mean[(s, WITNESS, 0.0)][q], mean[(s, WITNESS, THETA_TEST)][q]
            # Var of the readout-corrected mean: 1/c^2 - mu^2, not (1 - mu^2)/c^2
            c = 1 - 2 * P_RO
            var = max(1.0 / c ** 2 - (0.5 * (m0 + m1)) ** 2, 1e-9)
            vals.append(P.two_point_nstar(m0, m1, var))
        out[s] = np.array(vals)
        argmax_agrees[s] = agree
    return qs, out, argmax_agrees


def h3_t2(d):
    qs = d["config"]["qubits"]
    data = defaultdict(lambda: defaultdict(dict))
    for r in d["records"]:
        for bit, q in enumerate(qs):
            e, _ = P.expectation_from_counts(r["counts"], bit, P_RO)
            data[r["schedule"]][q][r["window_us"]] = e
    t2 = defaultdict(dict)
    for s in data:
        for q in qs:
            us = np.array(sorted(data[s][q]))
            y = np.array([data[s][q][u] for u in us])
            g = y > 0.05
            if g.sum() >= 3:
                b, _ = np.polyfit(us[g], np.log(y[g]), 1)
                if b < 0:
                    t2[s][q] = -1.0 / b
    return qs, t2, data


def fig_response_and_cost(h1, h2):
    qs, deb, raw, sigma = h1_slopes(h1)
    qs2, ns, argmax_agrees = h2_nstar(h2)
    for _s, _a in argmax_agrees.items():
        print("  %-10s kernel witness <%s> is the empirical argmax on %d/%d qubits"
              % (_s, WITNESS, _a, len(qs2)))

    fig, ax = plt.subplots(1, 2, figsize=figstyle.figsize(1.0, 2.6))

    # ---- (a) first-order response, measured vs the frozen prediction ----------
    order = ["free", "xy4_drop1", "xy4_asym", "xy4"]
    x = np.arange(len(order))
    meas = [np.median([deb[s][q] for q in qs]) for s in order]
    pre = [PRE_SLOPE[s] for s in order]
    lo = [np.percentile([deb[s][q] for q in qs], 25) for s in order]
    hi = [np.percentile([deb[s][q] for q in qs], 75) for s in order]
    err = np.abs(np.vstack([np.array(meas) - lo, np.array(hi) - np.array(meas)]))

    # 2-sigma resolution floor on the MEDIAN. The median of n normal draws has standard error
    # sqrt(pi/2) times the mean's, so the naive 2*sigma/sqrt(n) understates it; a percentile bootstrap over
    # the qubits is used instead and the analytic form is kept as a cross-check.
    _bs = np.random.default_rng(20260628)
    _draws = np.array([deb["xy4"][q] for q in qs], dtype=float)
    _meds = np.median(_bs.choice(_draws, size=(20000, len(_draws)), replace=True), axis=1)
    bound = float(np.percentile(np.abs(_meds), 97.5))
    bound_analytic = 2 * np.sqrt(np.pi / 2) * sigma / np.sqrt(len(qs))
    ax[0].bar(x - 0.2, pre, 0.4, label="predicted (simulation)", color="0.72", edgecolor="k", linewidth=0.4)
    ax[0].bar(x[:3] + 0.2, meas[:3], 0.4, yerr=err[:, :3], capsize=2, hatch="//",
              label="measured (device)", color="C0", edgecolor="k", linewidth=0.4)
    ax[0].errorbar([x[3] + 0.2], [bound], yerr=[[bound * 0.55], [0]], uplims=True,
                   fmt="v", color="C3", markersize=4, label="bootstrap 97.5th pct.")
    ax[0].set_yscale("log")
    ax[0].set_ylim(3e-3, 60)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([LABEL[s] for s in order], rotation=15, ha="right")
    ax[0].set_ylabel("first-order response $\\|\\partial s/\\partial\\theta\\|$")
    ax[0].axhline(bound, ls=":", c="0.4", lw=0.8)
    ax[0].legend(loc="lower left", fontsize=figstyle.ANNOT_PT)

    # ---- (b) detection cost per qubit ----------------------------------
    idx = np.arange(len(qs2))
    ax[1].axhspan(0.5 * PRE_NSTAR, 2 * PRE_NSTAR, color="C2", alpha=0.15,
                  label="pre-specified band")
    ax[1].axhline(PRE_NSTAR, ls="--", c="C2", lw=1.0, label="predicted $N^\\star$")
    ax[1].semilogy(idx, ns["xy4_asym"], "o", ms=3.4, color="C0", label="asym. XY4 (exposing edit)")
    ax[1].semilogy(idx, ns["xy4"], "s", ms=3.4, color="C3", mfc="none", label="XY4 (production)")
    ax[1].set_xlabel("qubit (20 measured in parallel)")
    ax[1].set_ylabel("detection cost $N^\\star$ (shots)")
    ax[1].set_xticks(idx[::4])
    ax[1].set_xticklabels([str(qs2[i]) for i in idx[::4]])
    ax[1].legend(loc="upper left", fontsize=figstyle.ANNOT_PT, ncol=1)

    fig.tight_layout()
    figstyle.save(fig, OUT, "fig_hw_response_cost", png_preview=False)
    return meas, pre, bound, ns


def fig_protection(h3):
    qs, t2, data = h3_t2(h3)
    fig, ax = plt.subplots(1, 2, figsize=figstyle.figsize(1.0, 2.4))

    # ---- (a) decay curves, median over qubits --------------------------
    styles = {"free": ("^-", "C3"), "xy4": ("o-", "C0"), "xy4_asym": ("s-", "C1")}
    for s in ("free", "xy4", "xy4_asym"):
        us = np.array(sorted(data[s][qs[0]]))
        med = np.array([np.median([data[s][q][u] for q in qs]) for u in us])
        mk, c = styles[s]
        ax[0].semilogx(us, med, mk, color=c, label=LABEL[s])
    ax[0].set_xlabel("sequence window $T$ ($\\mu$s)")
    ax[0].set_ylabel("$\\langle X\\rangle$ (median over qubits)")
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].legend(fontsize=figstyle.ANNOT_PT)

    # ---- (b) retention, per qubit --------------------------------------
    common = [q for q in qs if q in t2["xy4"] and q in t2["xy4_asym"]]
    ret = np.array([t2["xy4_asym"][q] / t2["xy4"][q] for q in common])
    rng = np.random.default_rng(20260628)
    boot = [np.median(rng.choice(ret, len(ret))) for _ in range(4000)]
    loci, hici = np.percentile(boot, [2.5, 97.5])
    ax[1].hist(ret, bins=np.linspace(0.2, 1.1, 13), color="C0", edgecolor="k", linewidth=0.4)
    ax[1].axvline(0.85, ls="--", c="C2", lw=1.2, label="fixed in advance: $\\geq 0.85$")
    ax[1].axvline(np.median(ret), ls="-", c="C3", lw=1.2,
                  label="measured median %.3f" % np.median(ret))
    ax[1].axvspan(loci, hici, color="C3", alpha=0.15, label="95% CI")
    ax[1].set_xlabel("protection retained, asym. XY4 / XY4")
    ax[1].set_ylabel("qubits")
    ax[1].legend(fontsize=figstyle.ANNOT_PT, loc="upper left")

    fig.tight_layout()
    figstyle.save(fig, OUT, "fig_hw_protection", png_preview=False)
    return t2, ret, (loci, hici)


def main():
    figstyle.apply()
    h1 = newest("h1_ibm_cleveland_*.json")
    h2 = newest("h2_ibm_cleveland_*.json")
    h3 = json.load(open(sorted(glob.glob(os.path.join(RES, "h3_ibm_cleveland_*.json")))[-1]))

    meas, pre, bound, ns = fig_response_and_cost(h1, h2)
    t2, ret, ci = fig_protection(h3)

    print("\n===== numbers used in the manuscript =====")
    for s, m, p in zip(["free", "xy4_drop1", "xy4_asym", "xy4"], meas, pre):
        print("  %-11s measured %8.3f   predicted %8.4g   ratio %s"
              % (s, m, p, "%.3f" % (m / p) if p > 1e-4 else "-"))
    print("  XY4 2-sigma upper limit on the median response: %.3f" % bound)
    print("  XY4 suppression lower bound                   : %.0fx" % (meas[0] / bound))
    print("  N* asym  median %8.0f  (predicted %.0f, ratio %.2f)"
          % (np.median(ns["xy4_asym"]), PRE_NSTAR, np.median(ns["xy4_asym"]) / PRE_NSTAR))
    print("  N* XY4   median %8.0f  (%.0fx more expensive)"
          % (np.median(ns["xy4"]), np.median(ns["xy4"]) / np.median(ns["xy4_asym"])))
    for s in ("free", "xy4", "xy4_asym"):
        v = list(t2[s].values())
        print("  T2 %-9s median %6.1f us" % (s, np.median(v)))
    print("  retention median %.3f  CI [%.3f, %.3f]" % (np.median(ret), *ci))
    print("  wrote fig_hw_response_cost.pdf and fig_hw_protection.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
