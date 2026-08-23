"""Preflight: inspect the backend, solve the timing grid, price the campaign.

Costs ZERO QPU time -- it reads calibration metadata and does arithmetic. Run it
first, every time, and especially after any backend change: `dt`, the alignment
granularity and the X-pulse duration all feed into whether the schedules can be
placed exactly, and an unplaceable schedule silently forges the signal we are
trying to measure.

    python preflight.py --fake                       # no credentials needed
    python preflight.py --backend ibm_cleveland      # the real thing

If the account is not configured yet, see README.md -- the CCQC instance details
have to come from the Miami Quantum Program Coordinator.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import pdet_hw as P

DEFAULT_WINDOW_S = 16e-6
DEFAULT_THETAS = (0.0, 0.025, 0.05, 0.10)      # rad/us; the sweep is run signed


def get_backend(args):
    if args.fake:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        return FakeSherbrooke(), "fake_sherbrooke (Eagle-class stand-in)"
    from qiskit_ibm_runtime import QiskitRuntimeService
    kwargs = {}
    if args.instance:
        kwargs["instance"] = args.instance
    if args.channel:
        kwargs["channel"] = args.channel
    service = QiskitRuntimeService(**kwargs)
    if args.backend:
        return service.backend(args.backend), args.backend
    b = service.least_busy(operational=True, simulator=False)
    return b, b.name


def _utcstamp():
    """UTC stamp in the same shape the measurement files use."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake", action="store_true", help="use a local Eagle-class stand-in")
    ap.add_argument("--backend", default=None, help="backend name (ask CECHelp for the CCQC one)")
    ap.add_argument("--instance", default=None, help="IBM instance / CRN for the CCQC allocation")
    ap.add_argument("--channel", default=None, help="runtime channel, e.g. ibm_quantum_platform")
    ap.add_argument("--window-us", type=float, default=DEFAULT_WINDOW_S * 1e6,
                    help="target DD window in microseconds (default 16)")
    ap.add_argument("--qubits", type=int, default=20, help="how many parallel qubits to use")
    ap.add_argument("--rep-delay-us", type=float, default=260.9,
                    help="per-shot overhead on top of the circuit. Default is MEASURED on "
                         "ibm_cleveland (job da3mqn6aa69c739irie0, 2026-08-20): 21 circuits x "
                         "1000 shots reported 6.2966 s of circuit execution, which after "
                         "subtracting the windows and the 2.652 us readout leaves 260.9 us per "
                         "shot -- within 5% of the backend's default_rep_delay of 250 us.")
    ap.add_argument("--allocation-min", type=float, default=30.0,
                    help="QPU minutes actually available this period (DQC methods = 30)")
    ap.add_argument("--json", default=None, help="write the resolved config here")
    args = ap.parse_args()

    backend, label = get_backend(args)
    print("=" * 72)
    print("BACKEND: %s" % label)
    print("=" * 72)
    print("  qubits          : %d" % backend.num_qubits)
    print("  basis gates     : %s" % ", ".join(sorted(backend.target.operation_names)))

    timing = P.timing_from_backend(backend, 0)
    print("  dt              : %.6f ns" % (timing.dt * 1e9))
    print("  alignment grid  : %d dt (%.3f ns)" % (timing.granularity,
                                                   timing.granularity * timing.dt * 1e9))
    print("  X duration      : %d dt (%.2f ns)" % (timing.x_dur, timing.x_dur * timing.dt * 1e9))

    has_pulse = any(n in backend.target.operation_names for n in ("pulse", "play"))
    print("  pulse-level     : %s" % ("exposed" if has_pulse
                                      else "not exposed (expected; the plan needs none)"))

    # ---- timing grid --------------------------------------------------
    print("\n" + "=" * 72)
    print("TIMING GRID")
    print("=" * 72)
    quantum = P.window_quantum(
        [f for name in P.SCHEDULES for f in P.SCHEDULES[name][0]], timing.granularity)
    window = P.solve_window(timing, args.window_us * 1e-6)
    print("  window quantum  : %d dt -- every usable window is a multiple of this" % quantum)
    print("  chosen window   : %d dt = %.4f us  (target %.1f us)"
          % (window, timing.to_us(window), args.window_us))
    if abs(timing.to_us(window) - args.window_us) > 0.5:
        print("  NOTE: the grid pulled the window %.3f us off target; this is fine as long as"
              % (timing.to_us(window) - args.window_us))
        print("        every schedule uses the SAME window, which it does.")

    print("\n  %-11s %-46s %s" % ("schedule", "free segments (dt)", "centre error"))
    ok = True
    layouts = {}
    for name in P.SCHEDULES:
        try:
            lay = P.layout_schedule(name, timing, window)
        except ValueError as exc:
            print("  %-11s UNPLACEABLE: %s" % (name, exc))
            ok = False
            continue
        layouts[name] = lay
        v = P.verify_layout(lay, timing)
        segs = ",".join(str(s) for s in lay.segments)
        if len(segs) > 44:
            segs = segs[:41] + "..."
        flag = "OK" if v["max_abs_error_dt"] == 0 and v["length_error_dt"] == 0 else "MISPLACED"
        if flag != "OK":
            ok = False
        print("  %-11s %-46s %s" % (name, segs, flag))

    if not ok:
        print("\n  ABORT: at least one schedule cannot be placed exactly on this grid.")
        print("  Re-run with a different --window-us; do not round the delays by hand.")
        return 1
    print("\n  All schedules place their pi-pulse centres exactly. Safe to build circuits.")

    # ---- qubit selection ----------------------------------------------
    print("\n" + "=" * 72)
    print("QUBIT SELECTION")
    print("=" * 72)
    qubits = P.pick_qubits(backend, args.qubits, min_distance=2)
    print("  %d qubits, pairwise >= 2 hops apart:" % len(qubits))
    print("  %s" % qubits)
    try:
        props = backend.properties()
        snap = {"backend": backend.name, "recorded_utc": _utcstamp(), "qubits": {}}
        print("\n  %-6s %10s %10s %12s" % ("qubit", "T1 (us)", "T2 (us)", "readout err"))
        for q in qubits:
            t1 = (props.t1(q) or 0.0) * 1e6
            t2 = (props.t2(q) or 0.0) * 1e6
            ro = props.readout_error(q)
            snap["qubits"][str(q)] = {"T1_us": t1, "T2_us": t2, "readout_error": ro}
            if q in qubits[:10]:
                print("  %-6d %10.1f %10.1f %12.4f" % (q, t1, t2, ro))
        if len(qubits) > 10:
            print("  ... %d more" % (len(qubits) - 10))
        # The manuscript quotes this per-qubit calibration, so it is written next to the measurement records
        # rather than left in a console scrollback where nobody can check it.
        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_hw")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "calibration_%s_%s.json" % (backend.name, snap["recorded_utc"]))
        with open(path, "w") as fh:
            json.dump(snap, fh, indent=2)
        print("\n  calibration snapshot written to %s" % path)
    except Exception as exc:
        print("  (calibration properties unavailable: %s)" % exc)

    # ---- budget --------------------------------------------------------
    print("\n" + "=" * 72)
    print("QPU BUDGET ESTIMATE")
    print("=" * 72)
    try:
        meas_s = backend.target["measure"][(qubits[0],)].duration or 1e-6
    except Exception:
        meas_s = 1e-6
    circuit_s = timing.to_seconds(window) + meas_s
    shot_s = circuit_s + args.rep_delay_us * 1e-6
    print("  circuit duration: %.2f us   (window %.2f + measure %.2f)"
          % (circuit_s * 1e6, timing.to_us(window), meas_s * 1e6))
    print("  per-shot time   : %.2f us   (+ %.0f us repetition delay)"
          % (shot_s * 1e6, args.rep_delay_us))
    print("  NOTE: the per-shot overhead dominates. The default is MEASURED on")
    print("        ibm_cleveland, not assumed; re-measure it if the backend changes.")

    plan = [
        ("H0  calibration",      len(qubits),          4_000),
        ("H1  blindness",        4 * 7 * 3,           20_000),
        ("H2  N* pool (1 day)",  2 * 2 * 3,          100_000),
        ("H3a noise spectrum",   6 * 8,                8_000),
        ("H3b coherence",        3 * 8,                8_000),
        ("H5  real error",       2 * 2 * 3,           20_000),
    ]
    total_s = 0.0
    print("\n  %-22s %10s %12s %12s" % ("experiment", "circuits", "shots/circ", "est. minutes"))
    for name, ncirc, nshot in plan:
        secs = ncirc * nshot * shot_s
        total_s += secs
        print("  %-22s %10d %12s %12.1f" % (name, ncirc, "{:,}".format(nshot), secs / 60))
    # H2's pool is re-acquired on three separate days so drift can be separated
    # from shot noise; the plan table above counts only the first day.
    h2_extra = 2 * plan[2][1] * plan[2][2] * shot_s
    total_s += h2_extra
    print("  %-22s %10s %12s %12.1f" % ("H2 days 2-3 (extra)", "", "", h2_extra / 60))
    print("  " + "-" * 58)
    print("  %-22s %10s %12s %12.1f" % ("TOTAL (one pass)", "", "", total_s / 60))
    print("  %-22s %10s %12s %12.1f" % ("with 3x contingency", "", "", 3 * total_s / 60))
    print("\n  All qubits run inside the same circuits, so per-qubit statistics (H4)")
    print("  cost nothing extra.")

    alloc = args.allocation_min
    print("\n  allocation this period : %.1f min" % alloc)
    print("  full plan, one pass    : %.1f min" % (total_s / 60))
    if total_s / 60 <= alloc:
        print("  -> fits, with %.1f min to spare." % (alloc - total_s / 60))
    else:
        print("  -> DOES NOT FIT. Cut scope in this order (the plan's own priority is")
        print("     H2 > H1 > H4 > H3 > H5, because H2 is the only experiment that tests")
        print("     whether the predicted cost was right):")
        print("       - drop H3a and H5                       saves %.1f min"
              % ((plan[3][1] * plan[3][2] + plan[5][1] * plan[5][2]) * shot_s / 60))
        print("       - H2 on one day instead of three        saves %.1f min" % (h2_extra / 60))
        print("       - H1 theta sweep 7 -> 5 points          saves %.1f min"
              % (2 * 4 * 3 * plan[1][2] * shot_s / 60))
        core = (plan[0][1] * plan[0][2] + (5 / 7) * plan[1][1] * plan[1][2]
                + plan[2][1] * plan[2][2] + plan[4][1] * plan[4][2]) * shot_s / 60
        print("     core campaign (H0 + reduced H1 + H2 + H3b): %.1f min" % core)
    print("\n  Times above are CIRCUIT EXECUTION. On the calibration job the metered charge")
    print("  ran 1.43x execution (9 s charged vs 6.30 s executed) -- a per-job load overhead")
    print("  that amortises over shot-heavy jobs, but there is only one data point behind it.")
    print("  Keep the committed plan under ~20 min of estimated execution to stay safe.")

    cfg = {
        "backend": label,
        "dt_s": timing.dt,
        "granularity_dt": timing.granularity,
        "x_dur_dt": timing.x_dur,
        "window_dt": window,
        "window_us": timing.to_us(window),
        "qubits": qubits,
        "rep_delay_us": args.rep_delay_us,
        "per_shot_s": shot_s,
        "schedules": {k: {"segments_dt": list(v.segments), "axes": list(v.axes)}
                      for k, v in layouts.items()},
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(cfg, fh, indent=2)
        print("\n  wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
