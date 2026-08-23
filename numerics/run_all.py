"""
Regenerate ALL PDET numerics + figures from scratch (reproducibility entry point).
Usage: python run_all.py   (seed 20260628 fixed in every module; ~minutes total)
Each module writes its artifacts under ../results/ ; this driver runs them in dependency order and reports
pass/fail. The A1 correctness gate (pdet_core self-test) runs FIRST and must pass before anything is trusted.
"""
import importlib, subprocess, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

MODULES = [
    ("pdet_core", "A1 correctness gate (response map vs exact propagation)"),
    ("run_phase0", "Phase-0 de-risk: kernel/gamma/eta2, baselines, Go/No-go"),
    ("phase1_knob", "Phase-1 readout-rotation probe (measurement design, not control)"),
    ("phase1b_dd_blindspot", "Phase-1 genuine control knob (DD blind spot)"),
    ("phase1c_filterfunction_baseline", "filter-function/DD-spectroscopy baseline"),
    ("phase1d_fisher_baseline", "Fisher/active experiment-design baseline"),
    ("a3_upper_bound_check", "A3 two-point finite-shot upper bound (tight)"),
    ("a4_kerA_characterization", "A4 ker A control-knob characterization"),
    ("phase4_evaluation", "Phase-4 evaluation suite (Table1 + Fig1-3, MC + CI)"),
    ("phase4b_benchmark", "R-O6: FWER discovery + equal-budget + protection Pareto"),
    ("phase4c_realism", "K_eff packing + operational invisibility + SPAM"),
    ("phase4d_noise_nuisance", "1/f filter-function protection + nuisance handling"),
    ("selfcheck_realistic_noise", "self-check L1: open-system T1/T2/readout noise"),
    ("selfcheck_dd_idle_usecase", "self-check: genuine non-contrived DD-on-idle use case"),
    ("selfcheck_baseline_headtohead", "self-check: equal-budget baseline head-to-head"),
    ("selfcheck_scalability", "self-check L6: locality-scaled scalability"),
    ("selfcheck_echoed_cr_usecase", "self-check: echoed-CR probe (which lever exposes the hidden ZI)"),
    ("selfcheck_model_mismatch", "self-check: model-mismatch sensitivity of the verdict"),
    ("kernel_factorization", "ker K vs ker M / ker K: the two obstructions, per dictionary"),
    ("pulsewidth_sensitivity", "how far the frozen predictions depend on the assumed pulse width"),
    ("fig_pulsewidth_suppression", "blind-axis suppression against pulse width (the finite-pulse figure)"),
    ("open_system_margin", "M_open vs M, covariance-aware cost, S(omega)-dependence of protection"),
    ("algorithm_constraints", "target-preserving edits; preparation-blind vs measurement-blind"),
    ("vz_vs_continuous", "virtual-Z frame injection vs a continuous physical detuning"),
    ("allocation_nuisance", "when a single witness setting is (and is not) the optimal allocation"),
    ("fig_2q_and_sensitivity", "2q CR per-direction margins + rank-tolerance sensitivity (paper figs)"),
    ("fig_detection_and_t2", "detection-error curves and the T2 sweep"),
    ("sync_paper_figures", "copy the manuscript-bound vector PDFs into paper/figures/"),
]

def run():
    results = []
    for mod, desc in MODULES:
        t0 = time.perf_counter()
        try:
            m = importlib.import_module(mod)
            if mod == "pdet_core":
                # the A1 self-test runs under __main__, so re-exec it and hold the driver to its exit status
                subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "pdet_core.py")],
                               check=True)
            else:
                m.main()
            ok = True; err = ""
        except Exception as e:
            ok = False; err = repr(e)
        dt = time.perf_counter() - t0
        results.append((mod, ok, round(dt, 1), err))
        print(f"[{'OK ' if ok else 'FAIL'}] {mod:34s} {dt:6.1f}s  {desc}" + (f"\n      ERROR: {err}" if not ok else ""))
    npass = sum(1 for _, ok, _, _ in results if ok)
    print(f"\n{npass}/{len(results)} modules regenerated successfully.")
    return results

if __name__ == "__main__":
    res = run()
    # a regeneration run that lost a module must not look like a success to a script or a CI job
    sys.exit(0 if all(ok for _, ok, _, _ in res) else 1)
