"""Analyse saved CCQC results against the predictions frozen before acquisition.

    python analyze.py results_hw/h1_ibm_xxx_2026....json
    python analyze.py results_hw/h2_*.json --readout-error 0.013

Reports per qubit, not pooled: the point of running 20 qubits at once is to show
the verdict is a property of the schedule rather than of one lucky qubit, so the
spread across qubits is a result and averaging it away would throw it out.

The analysis path here is the same one `validate_local.py` exercises on
simulated data, so a surprise in the numbers is a statement about the device
rather than about this file.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

import pdet_hw as P

THETA_TEST = 0.05

# The test on N*, fixed in advance, is a factor-of-2 band, so the search grid has to
# resolve better than a factor of 2 or the verdict is decided by the grid rather
# than by the device. ~25 points per three and a half decades gives ~1.4x steps.
N_GRID = tuple(sorted(set(int(round(x)) for x in np.logspace(2, 5.5, 25))))

# frozen before acquisition, from results/selfcheck/dd_idle_usecase_results.json (T=16 us,
# T1=200 us, T2=120 us, p_ro=1.3%). The hardware window differs slightly because
# it is snapped to the backend grid, so compare shapes and ratios, not digits.
PREREGISTERED = {
    "free":      {"slope": 14.00,  "nstar": 23.3},
    "xy4":       {"slope": 2e-5,   "nstar": 8.9e12},
    "xy4_drop1": {"slope": 3.33,   "nstar": 412.6},
    "xy4_asym":  {"slope": 0.700,  "nstar": 9310.4},
}


def load(paths: List[str]) -> List[Dict]:
    out = []
    for pattern in paths:
        for p in sorted(glob.glob(pattern)):
            with open(p) as fh:
                d = json.load(fh)
            d["_path"] = p
            out.append(d)
    if not out:
        raise SystemExit("no result files matched %s" % paths)
    return out


def counts_to_pool(counts: Dict[str, int], bit: int) -> np.ndarray:
    """Expand a counts dict into the per-shot +/-1 outcomes the bootstrap needs."""
    vals, reps = [], []
    for bitstring, n in counts.items():
        clean = bitstring.replace(" ", "")
        vals.append(1.0 if clean[::-1][bit] == "0" else -1.0)
        reps.append(n)
    return np.repeat(np.array(vals), np.array(reps))


def analyse_h1(payload: Dict, p_ro: float) -> None:
    cfg = payload["config"]
    qubits = cfg["qubits"]
    thetas = cfg["theta_sweep"]

    # signal[schedule][qubit][theta][basis]
    sig = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for rec in payload["records"]:
        for bit, q in enumerate(qubits):
            e, _ = P.expectation_from_counts(rec["counts"], bit, p_ro)
            sig[rec["schedule"]][q][rec["theta"]][rec["basis"]] = e

    print("\nH1 -- first-order response of full single-qubit tomography")
    print("   window %.4f us, %d qubits, readout correction p=%.4f"
          % (cfg["window_us"], len(qubits), p_ro))

    # The response vector's NORM is positively biased: three noisy components each
    # of variance sigma^2 give E[norm^2] = |true|^2 + 3 sigma^2, so pure noise still
    # reports a positive "signal". Subtract it. Projecting on <Y> instead is not a
    # fix -- a real qubit carries a static detuning, so on an unprotected schedule
    # the accumulated phase rotates the response out of Y and the projection then
    # under-reports. The debiased norm is right for both.
    slopes, debiased, sigmas = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for sched in sig:
        for q in qubits:
            rows = [[sig[sched][q][t][b] for b in P.BASES] for t in thetas]
            fit = P.signal_slope(thetas, rows)
            sig2 = float(np.mean(np.square(fit["slope_ses"])))
            slopes[sched][q] = fit["norm"]
            sigmas[sched][q] = np.sqrt(sig2)
            debiased[sched][q] = float(np.sqrt(max(fit["norm"] ** 2 - 3 * sig2, 0.0)))

    sig_typ = np.median([sigmas[s][q] for s in sigmas for q in qubits])
    print("\n   per-component slope std. error: %.3f  ->  a pure-noise vector still" % sig_typ)
    print("   reports a norm of about %.3f, so norms are debiased below." % (sig_typ * 1.6))

    print("\n   %-11s %10s %10s %10s %12s  %s"
          % ("schedule", "median", "min", "max", "pre-reg", "median/pre-reg"))
    for sched in ("free", "xy4_drop1", "xy4_asym", "xy4"):
        if sched not in debiased:
            continue
        v = np.array([debiased[sched][q] for q in qubits])
        pre = PREREGISTERED[sched]["slope"]
        ratio = "%.3f" % (np.median(v) / pre) if pre > 1e-4 else "-"
        print("   %-11s %10.4f %10.4f %10.4f %12.4g  %s"
              % (sched, np.median(v), v.min(), v.max(), pre, ratio))

    if "free" in debiased and "xy4" in debiased:
        raw = np.median([slopes["xy4"][q] for q in qubits])
        noise = sig_typ * 1.6
        free_med = np.median([debiased["free"][q] for q in qubits])
        se_med = sig_typ / np.sqrt(len(qubits))
        print("\n   XY4 blind spot:")
        print("     raw norm %.3f vs pure-noise expectation %.3f -> %s"
              % (raw, noise,
                 "CONSISTENT WITH ZERO" if raw <= noise else "residual above the noise floor"))
        print("     2-sigma lower bound on the suppression: %.0fx" % (free_med / (2 * se_med)))
        print("   The simulation's 2e-5 is the ideal-pulse limit. What hardware delivers at")
        print("   this shot budget is a BOUND, not a value: the first-order signal under XY4")
        print("   is not resolvable, and that bound is the honest number for remark 1.")


def analyse_h2(payload: Dict, p_ro: float) -> None:
    cfg = payload["config"]
    qubits = cfg["qubits"]

    pools = defaultdict(dict)      # pools[(schedule, basis, theta)][qubit] = +/-1 array
    means = defaultdict(dict)
    for rec in payload["records"]:
        key = (rec["schedule"], rec["basis"], rec["theta"])
        for bit, q in enumerate(qubits):
            pools[key][q] = counts_to_pool(rec["counts"], bit)
            e, _ = P.expectation_from_counts(rec["counts"], bit, p_ro)
            means[key][q] = e

    print("\nH2 -- two-point detection cost on the witness setting")
    print("   window %.4f us, theta = %.3f rad/us (%.2f kHz)"
          % (cfg["window_us"], THETA_TEST, THETA_TEST * 1e6 / (2 * np.pi) / 1e3))

    for sched in ("xy4_asym", "xy4"):
        rows = []
        for q in qubits:
            diffs = {}
            for b in P.BASES:
                k0, k1 = (sched, b, 0.0), (sched, b, THETA_TEST)
                if q not in means.get(k0, {}) or q not in means.get(k1, {}):
                    continue
                diffs[b] = means[k1][q] - means[k0][q]
            if not diffs:
                continue
            # the workflow prescribes the witness. Taking the empirical argmax of three noisy
            # separations instead biases |gamma| away from zero, which on a schedule whose response
            # is not resolved manufactures a finite, acquisition-size-dependent N*.
            wb = P.WITNESS
            if wb not in diffs:
                continue
            m0 = means[(sched, wb, 0.0)][q]
            m1 = means[(sched, wb, THETA_TEST)][q]
            mbar = 0.5 * (m0 + m1)
            # readout-corrected estimator mu = m_raw / c with c = 1 - 2 p_ro, so
            # Var(mu) = Var(m_raw)/c^2 = (1 - m_raw^2)/c^2 = 1/c^2 - mu^2.
            c = 1 - 2 * p_ro
            var = max(1.0 / c ** 2 - mbar ** 2, 1e-9)
            n_an = P.two_point_nstar(m0, m1, var)
            rates = P.bootstrap_detection_rates(pools[(sched, wb, 0.0)][q],
                                                pools[(sched, wb, THETA_TEST)][q],
                                                N_GRID, n_trials=2000)
            n_em = P.nstar_from_rates(rates)
            rows.append((q, wb, abs(m1 - m0), n_an, n_em))

        if not rows:
            continue
        pre = PREREGISTERED[sched]["nstar"]
        print("\n   schedule %s   (frozen N* = %.4g)" % (sched, pre))
        print("   %-6s %7s %10s %14s %14s %10s"
              % ("qubit", "witness", "gamma", "N* analytic", "N* empirical", "obs/pred"))
        for q, wb, g, na, ne in rows:
            r = "-" if not np.isfinite(ne) or pre > 1e6 else "%.2f" % (ne / pre)
            print("   %-6d %7s %10.5f %14.0f %14s %10s"
                  % (q, wb, g, na, "inf" if not np.isfinite(ne) else "%.0f" % ne, r))
        ratios = np.array([ne / pre for _, _, _, _, ne in rows])      # inf where the grid never crosses
        n_all = len(ratios)
        n_cens = int(np.sum(~np.isfinite(ratios)))
        if n_all and pre < 1e6:
            fin = ratios[np.isfinite(ratios)]
            # report BOTH: the median over the qubits that cross, and the median over all qubits with
            # the non-crossings kept as censored above the grid. Dropping them changes the denominator.
            print("   crossings: %d of %d qubits reach the target rates on the search grid"
                  % (n_all - n_cens, n_all))
            if len(fin):
                print("   median observed/predicted, crossing qubits = %.2f   (band fixed in advance: 0.5-2)"
                      % np.median(fin))
            med_all = np.median(np.sort(ratios))
            print("   median observed/predicted, all %d with non-crossings censored = %s"
                  % (n_all, "inf" if not np.isfinite(med_all) else "%.2f" % med_all))
            frac_cross = float(np.mean((fin >= 0.3) & (fin <= 3))) if len(fin) else 0.0
            frac_all = (np.sum((fin >= 0.3) & (fin <= 3)) / n_all) if n_all else 0.0
            print("   inside [0.3, 3]: %.0f%% of the %d crossing qubits, %.0f%% of all %d"
                  % (100 * frac_cross, len(fin), 100 * frac_all, n_all))


def analyse_h3(payload: Dict, p_ro: float) -> None:
    cfg = payload["config"]
    qubits = cfg["qubits"]
    data = defaultdict(lambda: defaultdict(dict))
    for rec in payload["records"]:
        for bit, q in enumerate(qubits):
            e, _ = P.expectation_from_counts(rec["counts"], bit, p_ro)
            data[rec["schedule"]][q][rec["window_us"]] = e

    print("\nH3 -- coherence under each schedule (no injected drift)")

    # Fit T2 per qubit per schedule, then form the ratio PER QUBIT. Taking a ratio
    # of medians instead of the median of ratios is not the same statistic and can
    # move the answer across a decision threshold -- it did here, by 0.04.
    per_qubit = {}
    for sched in data:
        vals = {}
        for q in qubits:
            us = np.array(sorted(data[sched][q]))
            y = np.array([data[sched][q][u] for u in us])
            good = y > 0.05
            if good.sum() < 3:
                continue
            b, _a = np.polyfit(us[good], np.log(y[good]), 1)   # <X> = exp(-t/T2)
            if b < 0:
                vals[q] = -1.0 / b
        per_qubit[sched] = vals

    t2 = {s: (np.median(list(v.values())) if v else float("nan"))
          for s, v in per_qubit.items()}
    base = t2.get("free", float("nan"))
    print("\n   %-11s %14s %14s" % ("schedule", "median T2seq (us)", "vs free"))
    for sched in ("free", "xy4", "xy4_asym"):
        if sched in t2:
            print("   %-11s %14.1f %14.2f" % (sched, t2[sched], t2[sched] / base))

    if "xy4" in per_qubit and "xy4_asym" in per_qubit:
        common = [q for q in qubits
                  if q in per_qubit["xy4"] and q in per_qubit["xy4_asym"]]
        ratios = np.array([per_qubit["xy4_asym"][q] / per_qubit["xy4"][q] for q in common])
        ret = float(np.median(ratios))
        rng = np.random.default_rng(20260628)
        boot = [np.median(rng.choice(ratios, len(ratios))) for _ in range(4000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print("\n   retention (asym / XY4), per qubit then median = %.3f" % ret)
        print("     95%% CI [%.3f, %.3f] over %d qubits; IQR %.3f-%.3f"
              % (lo, hi, len(common), *np.percentile(ratios, [25, 75])))
        print("     %.0f%% of qubits individually >= 0.85" % (100 * np.mean(ratios >= 0.85)))
        verdict = ("consistent with the threshold (0.85 lies inside the CI)"
                   if lo <= 0.85 <= hi else
                   "CLEARS the threshold" if lo > 0.85 else "MISSES the threshold")
        print("     vs the >= 0.85 fixed in advance: %s" % verdict)
        print("   fixed in advance: >= 0.85, and the ordering XY4 > asym > free must hold at 3 sigma.")
        print("   The paper's 0.93 is a filter-function ratio over an ASSUMED 1/f spectrum;")
        print("   this is the measured coherence ratio and is the number to quote.")
        if base and t2["xy4"] / base < 1.3:
            print("\n   WARNING: XY4 barely beats free evolution here. On a SIMULATED run that is")
            print("   expected and means nothing -- Aer's noise model is Markovian, and dynamical")
            print("   decoupling suppresses correlated (1/f) noise, which the model does not")
            print("   contain. On real hardware a ratio this small would instead be a real")
            print("   negative result about the device or the sequence, so check `provenance.simulated`")
            print("   before reading anything into it.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="result JSON files (globs allowed)")
    ap.add_argument("--readout-error", type=float, default=0.0,
                    help="symmetric readout error to correct for (from H0)")
    args = ap.parse_args()

    for payload in load(args.paths):
        exp = payload["config"]["experiment"]
        print("=" * 72)
        print("%s   %s   backend=%s   git=%s"
              % (payload["_path"], exp, payload["provenance"]["backend"],
                 payload["provenance"]["git_commit"][:10]))
        print("=" * 72)
        {"h1": analyse_h1, "h2": analyse_h2, "h3": analyse_h3}[exp](payload, args.readout_error)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
