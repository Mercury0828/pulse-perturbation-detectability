"""Build and submit the PDET hardware experiments to CCQC.

    python run_experiment.py h1 --fake                 # build only, no credentials
    python run_experiment.py h1 --backend ibm_xxx --dry-run
    python run_experiment.py h1 --backend ibm_xxx --submit

Nothing is sent until `--submit` is passed. `--dry-run` (the default) builds
every circuit, transpiles it, verifies the schedule survived transpilation, and
prints the shot and time cost, so the whole campaign can be rehearsed for free.

Raw counts are written to JSON with a provenance block -- backend name, the
calibration snapshot, the resolved timing grid, the git commit -- because the
pre-registration is only worth something if the run that produced a number can
be identified afterwards.

Experiments:
  h1  blindness: signed theta sweep x 4 schedules x 3 bases
  h2  N* pool:   two hypotheses x two schedules x 3 bases, interleaved, deep
  h3  coherence: window sweep, no injected drift
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
from qiskit import transpile

import pdet_hw as P

# H1 fits a FIRST-ORDER slope, so the sweep must stay in the linear regime. The
# unprotected signal is <Y> = -sin(theta*T); with T = 16 us, theta = 0.10 rad/us
# gives 1.6 rad, where a straight-line fit recovers only 0.69 of the true slope
# and the "damping" that comes out is an artefact of the fit, not the device.
# This sweep holds |theta*T| <= 0.16 rad (under 0.5% nonlinearity).
THETA_SWEEP = (-0.01, -0.005, -0.0025, 0.0, 0.0025, 0.005, 0.01)   # rad/us
THETA_TEST = 0.05        # H2's working point; a two-point test needs no linearity
H1_SCHEDULES = ("free", "xy4", "xy4_drop1", "xy4_asym")
H2_SCHEDULES = ("xy4", "xy4_asym")
H3_SCHEDULES = ("free", "xy4", "xy4_asym")
H3_WINDOWS_US = (2, 4, 8, 16, 32, 64, 128)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def get_backend(args):
    if args.fake:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        return FakeSherbrooke(), None
    from qiskit_ibm_runtime import QiskitRuntimeService
    kwargs = {}
    if args.instance:
        kwargs["instance"] = args.instance
    if args.channel:
        kwargs["channel"] = args.channel
    service = QiskitRuntimeService(**kwargs)
    return service.backend(args.backend), service


def build_h1(timing, window, qubits, nq) -> List[Dict]:
    jobs = []
    for name in H1_SCHEDULES:
        lay = P.layout_schedule(name, timing, window)
        for theta in THETA_SWEEP:
            for basis in P.BASES:
                jobs.append({"tag": "h1", "schedule": name, "theta": theta, "basis": basis,
                             "circuit": P.build_circuit(lay, timing, qubits, theta, basis, nq)})
    return jobs


def build_h2(timing, window, qubits, nq) -> List[Dict]:
    """Interleave the hypotheses so device drift cannot masquerade as the signal.

    Ordering matters more here than anywhere else in the campaign: if all the
    theta=0 circuits ran before all the theta=theta0 circuits, a slow frequency
    drift between the two halves would be indistinguishable from the injected
    detuning, and the measured N* would be meaningless.
    """
    jobs = []
    for basis in P.BASES:
        for name in H2_SCHEDULES:
            lay = P.layout_schedule(name, timing, window)
            for theta in (0.0, THETA_TEST):
                jobs.append({"tag": "h2", "schedule": name, "theta": theta, "basis": basis,
                             "circuit": P.build_circuit(lay, timing, qubits, theta, basis, nq)})
    # alternate hypotheses within the submitted order
    jobs.sort(key=lambda j: (j["basis"], j["schedule"]))
    out = []
    h0 = [j for j in jobs if j["theta"] == 0.0]
    h1 = [j for j in jobs if j["theta"] != 0.0]
    for a, b in zip(h0, h1):
        out.extend([a, b])
    return out


def build_h3(timing, qubits, nq) -> List[Dict]:
    jobs = []
    for us in H3_WINDOWS_US:
        w = P.solve_window(timing, us * 1e-6)
        for name in H3_SCHEDULES:
            lay = P.layout_schedule(name, timing, w)
            jobs.append({"tag": "h3", "schedule": name, "theta": 0.0, "basis": "X",
                         "window_dt": w, "window_us": timing.to_us(w),
                         "circuit": P.build_circuit(lay, timing, qubits, 0.0, "X", nq)})
    return jobs


def verify_transpiled(tqc, timing, expect_delay_dt: int, tol_dt: int = 0) -> Dict:
    """Confirm the transpiler did not reschedule, merge or drop our delays.

    optimization_level=0 should leave them alone, but "should" is not a check,
    and a silently shortened delay changes the physics rather than the runtime.
    """
    total = 0
    n_delay = 0
    for inst in tqc.data:
        if inst.operation.name == "delay":
            total += int(inst.operation.duration)
            n_delay += 1
    per_qubit = total / max(len(set(q for inst in tqc.data if inst.operation.name == "delay"
                                    for q in inst.qubits)), 1)
    ok = abs(per_qubit - expect_delay_dt) <= tol_dt
    return {"n_delay_instructions": n_delay, "delay_dt_per_qubit": per_qubit,
            "expected_dt": expect_delay_dt, "ok": bool(ok)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=("h1", "h2", "h3"))
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--instance", default=None)
    ap.add_argument("--channel", default=None)
    ap.add_argument("--window-us", type=float, default=16.0)
    ap.add_argument("--qubits", type=int, default=20)
    ap.add_argument("--qubit-list", default=None,
                    help="comma-separated physical qubits; overrides --qubits")
    ap.add_argument("--shots", type=int, default=None)
    ap.add_argument("--submit", action="store_true", help="actually send the job")
    ap.add_argument("--simulate", action="store_true",
                    help="run on Aer with the backend's noise model and write a result "
                         "file in the SAME format, so the analysis can be rehearsed for free")
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--out", default="results_hw")
    ap.add_argument("--seed", type=int, default=20260628)
    args = ap.parse_args()

    if args.submit and args.fake:
        print("refusing to --submit against a fake backend"); return 2
    if not args.fake and not args.backend:
        print("give --backend (ask CECHelp for the CCQC backend name) or use --fake"); return 2

    backend, service = get_backend(args)
    timing = P.timing_from_backend(backend, 0)
    window = P.solve_window(timing, args.window_us * 1e-6)
    nq = backend.num_qubits
    qubits = ([int(x) for x in args.qubit_list.split(",")] if args.qubit_list
              else P.pick_qubits(backend, args.qubits, min_distance=2))

    default_shots = {"h1": 20_000, "h2": 100_000, "h3": 8_000}
    shots = args.shots or default_shots[args.experiment]

    print("backend  : %s" % backend.name)
    print("window   : %d dt = %.4f us" % (window, timing.to_us(window)))
    print("qubits   : %s" % qubits)
    print("shots    : %s per circuit" % "{:,}".format(shots))

    builder = {"h1": lambda: build_h1(timing, window, qubits, nq),
               "h2": lambda: build_h2(timing, window, qubits, nq),
               "h3": lambda: build_h3(timing, qubits, nq)}[args.experiment]
    jobs = builder()
    print("circuits : %d" % len(jobs))

    # transpile and check the schedule survived
    layout = list(range(nq))
    tcircs = []
    bad = 0
    for j in jobs:
        tqc = transpile(j["circuit"], backend, optimization_level=0, initial_layout=layout)
        expect = j.get("window_dt", window)
        lay = P.layout_schedule(j["schedule"], timing, expect)
        chk = verify_transpiled(tqc, timing, lay.free_dt)
        if not chk["ok"]:
            bad += 1
            print("  !! %s theta=%s %s: delays changed (%s dt vs %s expected)"
                  % (j["schedule"], j["theta"], j["basis"],
                     chk["delay_dt_per_qubit"], chk["expected_dt"]))
        tcircs.append(tqc)
    if bad:
        print("ABORT: %d circuit(s) lost their schedule in transpilation." % bad)
        return 1
    print("transpile: all %d circuits kept their delays exactly" % len(jobs))

    try:
        meas_s = backend.target["measure"][(qubits[0],)].duration or 1e-6
    except Exception:
        meas_s = 1e-6
    per_shot = timing.to_seconds(window) + meas_s + 300e-6
    est_min = len(jobs) * shots * per_shot / 60
    print("estimate : %.1f QPU-minutes at a 300 us repetition delay" % est_min)
    timing_record = None

    if not args.submit and not args.simulate:
        print("\nDRY RUN -- nothing submitted. Re-run with --submit to send it,")
        print("or --simulate to rehearse the analysis on Aer at no QPU cost.")
        return 0

    if args.simulate:
        from qiskit_aer import AerSimulator
        sim = AerSimulator.from_backend(backend)
        rng = np.random.default_rng(args.seed)
        print("\nsimulating on Aer (backend noise model) ...")
        all_counts = []
        for tqc in tcircs:
            # a fresh seed per circuit: one shared seed makes two circuits with
            # identical distributions return identical counts, which would erase
            # exactly the shot noise this rehearsal is meant to exercise
            res = sim.run(tqc, shots=shots,
                          seed_simulator=int(rng.integers(1, 2 ** 31))).result()
            all_counts.append(res.get_counts())
        job_id = "aer-simulated"
    else:
        from qiskit_ibm_runtime import SamplerV2
        sampler = SamplerV2(mode=backend)

        # The metered cost is the one number no amount of arithmetic can predict,
        # so record it on every run: the allocation counter before and after, and
        # whatever the job itself reports.
        usage_before = None
        try:
            usage_before = service.usage() if service else None
        except Exception as exc:
            print("  (usage before unavailable: %s)" % exc)

        print("\nsubmitting ...")
        t_submit = time.time()
        job = sampler.run(tcircs, shots=shots)
        job_id = job.job_id()
        print("job id: %s" % job_id)
        print("waiting for the result ...")
        result = job.result()
        wall_s = time.time() - t_submit

        metrics = {}
        try:
            metrics = job.metrics() or {}
        except Exception as exc:
            print("  (job metrics unavailable: %s)" % exc)
        usage_after = None
        try:
            usage_after = service.usage() if service else None
        except Exception as exc:
            print("  (usage after unavailable: %s)" % exc)

        quantum_s = (metrics.get("usage", {}) or {}).get("quantum_seconds")
        total_shots = len(tcircs) * shots
        print("\n  wall clock (submit -> result): %.1f s" % wall_s)
        if quantum_s:
            print("  metered quantum time         : %.3f s" % quantum_s)
            print("  -> %.1f us per shot over %s shots"
                  % (quantum_s / total_shots * 1e6, "{:,}".format(total_shots)))
            print("  (the estimate above assumed %.1f us/shot)" % (per_shot * 1e6))
        if usage_before and usage_after:
            spent = (usage_before.get("usage_remaining_seconds", 0)
                     - usage_after.get("usage_remaining_seconds", 0))
            print("  allocation consumed          : %.1f s (%.1f s remaining)"
                  % (spent, usage_after.get("usage_remaining_seconds", -1)))
        timing_record = {"wall_seconds": wall_s, "metrics": metrics,
                         "usage_before": usage_before, "usage_after": usage_after,
                         "total_shots": total_shots,
                         "estimated_per_shot_s": per_shot}
        all_counts = []
        for pub in result:
            data = pub.data
            reg = "c" if hasattr(data, "c") else list(data.keys())[0]
            all_counts.append(getattr(data, reg).get_counts() if hasattr(data, reg)
                              else data[reg].get_counts())

    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    tag = "sim" if args.simulate else backend.name
    path = os.path.join(args.out, "%s_%s_%s.json" % (args.experiment, tag, stamp))

    records = []
    for j, counts in zip(jobs, all_counts):
        rec = {k: j[k] for k in ("tag", "schedule", "theta", "basis") if k in j}
        rec.update({k: j[k] for k in ("window_dt", "window_us") if k in j})
        rec["counts"] = counts
        records.append(rec)

    payload = {
        "provenance": {
            "git_commit": git_sha(),
            "backend": backend.name,
            "submitted_utc": stamp,
            "job_id": job_id,
            "simulated": bool(args.simulate),
            "timing": timing_record,
        },
        "config": {
            "experiment": args.experiment,
            "window_dt": window,
            "window_us": timing.to_us(window),
            "dt_s": timing.dt,
            "granularity_dt": timing.granularity,
            "x_dur_dt": timing.x_dur,
            "qubits": qubits,
            "shots": shots,
            "theta_sweep": list(THETA_SWEEP),
            "theta_test": THETA_TEST,
        },
        "records": records,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
