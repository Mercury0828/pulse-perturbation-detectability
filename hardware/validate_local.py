"""Validate the whole measurement and analysis chain locally, before spending QPU time.

Two levels:

  ideal  -- exact statevector, no noise. Checks the phase bookkeeping: the
            injected drift must cancel under XY4 and must survive, at the
            analytically known strength, under the asymmetric variant. The
            expected values are not fitted; they follow from the toggling-frame
            integral |int y(t) dt|, which is 0 for symmetric XY4, T/4 for
            XY4 drop-1 and T/20 for the asymmetric variant.

  noisy  -- Aer with the backend's own T1/T2/readout model, finite shots, and
            the real analysis path: counts -> expectation -> slope fit, and the
            bootstrap that produces N*. This is what catches an analysis bug
            that the ideal check cannot see.

    python validate_local.py --level ideal
    python validate_local.py --level noisy --shots 40000

A failure here is a bug in this repository, not a discovery. Fix it before
submitting anything to hardware.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Pauli, Statevector

import pdet_hw as P

# analytic |int y(t) dt| / T for each schedule -- the ideal first-order response
TOGGLING_INTEGRAL = {
    "free": 1.0,
    "echo": 0.0,
    "xy4": 0.0,
    "xy4_drop1": 0.25,
    "xy4_asym": 0.05,
}

# H1 measures a FIRST-ORDER response, so the sweep has to stay inside the linear
# regime. The signal on an unprotected qubit is <Y> = -sin(theta*T), so the
# linearisation error is theta*T squared over six: at the paper's working point
# theta=0.05 over T=16 us that is 0.8 rad and already 10% off, and at 0.10 it is
# 1.6 rad, where a straight-line fit recovers only 0.69 of the true slope. The
# sweep below keeps |theta*T| <= 0.16 rad, i.e. under 0.5% nonlinearity.
THETA_SWEEP = (-0.01, -0.005, -0.0025, 0.0, 0.0025, 0.005, 0.01)   # rad/us, signed
THETA_TEST = 0.05        # H2's working point; a two-point test needs no linearity

# fail the sweep if the unprotected phase leaves the linear regime
MAX_LINEAR_PHASE_RAD = 0.3


def ideal_signal(name, timing, window, theta):
    lay = P.layout_schedule(name, timing, window)
    qc = P.build_circuit(lay, timing, [0], theta, "Z", n_qubits_total=1)
    qc.remove_final_measurements(inplace=True)
    sv = Statevector.from_instruction(qc)
    return np.array([sv.expectation_value(Pauli(p)).real for p in P.BASES])


def run_ideal(timing, window) -> int:
    T_us = timing.to_us(window)
    print("\nIDEAL (exact statevector, ideal pulses, no decoherence)")
    print("  window T = %.4f us\n" % T_us)
    print("  %-11s %14s %14s %10s  %s" % ("schedule", "|ds/dtheta|", "analytic", "ratio", ""))
    eps = 1e-3
    bad = 0
    for name, frac in TOGGLING_INTEGRAL.items():
        d = (ideal_signal(name, timing, window, +eps)
             - ideal_signal(name, timing, window, -eps)) / (2 * eps)
        got = float(np.linalg.norm(d))
        want = frac * T_us
        if want == 0:
            ok = got < 1e-9
            print("  %-11s %14.3e %14.6f %10s  %s"
                  % (name, got, want, "n/a", "OK (blind)" if ok else "FAIL: signal leaked"))
        else:
            ok = abs(got / want - 1.0) < 1e-3
            print("  %-11s %14.6f %14.6f %10.6f  %s"
                  % (name, got, want, got / want, "OK" if ok else "FAIL"))
        bad += 0 if ok else 1
    return bad


def run_noisy(timing, window, backend, shots, seed) -> int:
    from qiskit_aer import AerSimulator

    sim = AerSimulator.from_backend(backend)
    T_us = timing.to_us(window)
    rng = np.random.default_rng(seed)
    nq = backend.num_qubits
    layout = list(range(nq))

    def measure(name, theta, tag):
        """Full path: build -> transpile -> shots -> counts -> expectation."""
        lay = P.layout_schedule(name, timing, window)
        out = []
        for basis in P.BASES:
            qc = P.build_circuit(lay, timing, [0], theta, basis, n_qubits_total=nq)
            tqc = transpile(qc, sim, optimization_level=0, initial_layout=layout)
            # a fresh seed per circuit: reusing one seed makes two circuits with
            # identical distributions return identical counts, which would hide
            # shot noise exactly where we are trying to measure it
            res = sim.run(tqc, shots=shots,
                          seed_simulator=int(rng.integers(1, 2 ** 31))).result()
            e, _ = P.expectation_from_counts(res.get_counts(), 0)
            out.append(e)
        return np.array(out)

    print("\nNOISY (Aer from_backend: T1/T2/readout, %d shots per setting)" % shots)
    print("  window T = %.4f us" % T_us)

    phase = max(abs(t) for t in THETA_SWEEP) * T_us
    nl = phase ** 2 / 6.0
    print("  sweep     : |theta|max = %.4f rad/us -> |theta*T| = %.3f rad, "
          "linearisation error %.2f%%" % (max(abs(t) for t in THETA_SWEEP), phase, 100 * nl))
    if phase > MAX_LINEAR_PHASE_RAD:
        print("  FAIL: the sweep leaves the linear regime; a straight-line fit would")
        print("        under-report the first-order slope by roughly %.0f%%." % (100 * nl))
        return 1
    print()

    # --- H1: signed sweep, slope fit ---------------------------------
    print("  H1 -- signed theta sweep, slope of the tomography signal\n")
    print("  %-11s %16s %12s %12s  %s"
          % ("schedule", "|ds/dtheta|+/-se", "ideal", "damping", "verdict"))
    results = {}
    bad = 0
    for name in ("free", "xy4", "xy4_asym"):
        rows = [measure(name, th, name) for th in THETA_SWEEP]
        fit = P.signal_slope(THETA_SWEEP, rows)
        got, se = fit["norm"], fit["norm_se"]
        ideal = TOGGLING_INTEGRAL[name] * T_us
        results[name] = fit
        if ideal == 0:
            # the blind spot must stay consistent with zero AND be strongly
            # suppressed relative to free evolution
            supp = results["free"]["norm"] / got if got > 0 else float("inf")
            ok = supp > 30
            print("  %-11s %8.4f+/-%.4f %12.4f %12s  %s"
                  % (name, got, se, ideal, "-",
                     "OK (suppressed %.0fx)" % supp if ok else "FAIL (only %.0fx)" % supp))
        else:
            # The measured slope is the ideal one times a damping factor that
            # depends on the device, so the test is whether the measurement is
            # CONSISTENT with a plausible damping -- comparing point estimates to
            # a hard band ignores the error bar and fails spuriously as soon as
            # the sweep narrows and the slope precision drops.
            lo, hi = 0.5 * ideal, 1.2 * ideal
            ok = (got + 2 * se) >= lo and (got - 2 * se) <= hi
            print("  %-11s %8.4f+/-%.4f %12.4f %12.3f  %s"
                  % (name, got, se, ideal, got / ideal,
                     "OK" if ok else "FAIL (2-sigma interval misses [%.2f, %.2f])" % (lo, hi)))
        bad += 0 if ok else 1

    # --- H2: witness setting, N* ---------------------------------------
    print("\n  H2 -- two-point cost on the single witness setting\n")
    s0 = measure("xy4_asym", 0.0, "h0")
    s1 = measure("xy4_asym", THETA_TEST, "h1")
    diff = np.abs(s1 - s0)
    wi = int(np.argmax(diff))
    gamma = float(diff[wi])
    print("  witness basis   : %s   (|s1-s0| = %.5f)" % (P.BASES[wi], gamma))

    # per-shot variance of a +/-1 observable, at the midpoint of the hypotheses
    mbar = 0.5 * (s0[wi] + s1[wi])
    var = max(1.0 - mbar ** 2, 1e-9)
    nstar_analytic = P.two_point_nstar(s0[wi], s1[wi], var)
    print("  per-shot var    : %.4f" % var)
    print("  N* (analytic)   : %.0f shots" % nstar_analytic)

    # empirical: synthesise the two shot pools from the measured means, then run
    # the same bootstrap the hardware analysis will run
    pool = 400_000
    p0 = 0.5 * (1 + s0[wi])
    p1 = 0.5 * (1 + s1[wi])
    sh0 = np.where(rng.random(pool) < p0, 1.0, -1.0)
    sh1 = np.where(rng.random(pool) < p1, 1.0, -1.0)
    grid = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
    rates = P.bootstrap_detection_rates(sh0, sh1, grid, n_trials=2000, seed=seed)
    print("\n  %10s %14s %10s" % ("N", "false alarm", "miss"))
    for r in rates:
        print("  %10d %14.4f %10.4f" % (r["N"], r["false_alarm"], r["miss"]))
    nstar_emp = P.nstar_from_rates(rates)
    print("\n  N* (empirical, on this grid): %s" % ("inf" if nstar_emp == float("inf")
                                                    else "%.0f" % nstar_emp))
    ratio = nstar_emp / nstar_analytic if np.isfinite(nstar_emp) else float("inf")
    ok = 0.2 < ratio < 5
    print("  empirical / analytic = %.2f  %s"
          % (ratio, "OK (same order, grid is coarse)" if ok else "FAIL"))
    bad += 0 if ok else 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", choices=("ideal", "noisy", "both"), default="both")
    ap.add_argument("--window-us", type=float, default=16.0)
    ap.add_argument("--shots", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=20260628)
    ap.add_argument("--backend", default=None,
                    help="validate against a REAL backend's calibration (e.g. ibm_cleveland). "
                         "Reads properties only -- no QPU time is consumed.")
    ap.add_argument("--qubit", type=int, default=0, help="physical qubit to model")
    args = ap.parse_args()

    if args.backend:
        from qiskit_ibm_runtime import QiskitRuntimeService
        backend = QiskitRuntimeService().backend(args.backend)
        print("validating against the live calibration of %s (no QPU time used)" % backend.name)
    else:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        backend = FakeSherbrooke()
    timing = P.timing_from_backend(backend, args.qubit)
    window = P.solve_window(timing, args.window_us * 1e-6)

    bad = 0
    if args.level in ("ideal", "both"):
        bad += run_ideal(timing, window)
    if args.level in ("noisy", "both"):
        bad += run_noisy(timing, window, backend, args.shots, args.seed)

    print("\n" + ("ALL CHECKS PASSED" if bad == 0 else "%d CHECK(S) FAILED" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
