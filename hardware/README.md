# PDET on `ibm_cleveland` — the device campaign

Everything runs at the **schedule level** — delays, X/Y pulses, virtual-Z frame updates. No pulse-level access
is used or needed, which matters because Qiskit Pulse was removed in Qiskit 2.0 and IBM backends no longer
expose it.

The measured results are in [`RESULTS.md`](RESULTS.md). The raw counts are in `results_hw/`, so the analysis and
the paper's device figures regenerate without an allocation.

## Regenerating the analysis and the figures (no device access)

```bash
python analyze.py "results_hw/h2_ibm_cleveland_*.json" --readout-error 0.006
python analyze.py "results_hw/h1_ibm_cleveland_*.json" --readout-error 0.006
python analyze.py "results_hw/h3_ibm_cleveland_*.json" --readout-error 0.006
python holdout_nstar.py            # threshold estimated from a disjoint calibration split
python make_paper_figures.py
```

Raw counts, the resolved timing grid, the qubit set, the git commit and the charged processor time are all in
the provenance block of each `results_hw/*.json`.

## Re-acquiring the data

Needs an account with an allocation on the backend.

```bash
pip install "qiskit>=2.0" qiskit-ibm-runtime qiskit-aer numpy scipy
```

Verified against qiskit 2.5.2 / qiskit-ibm-runtime 0.49.0 / qiskit-aer 0.17.2.

```bash
python connect_check.py --save-token <api-key>   # once; lists instances, allocation and backends
python preflight.py --backend ibm_cleveland      # grid solve, qubit selection, budget. Zero QPU cost
python validate_local.py --backend ibm_cleveland # ideal + noisy end-to-end check. Zero QPU cost
python run_experiment.py h1 --backend ibm_cleveland --qubits 20            # dry run (default)
python run_experiment.py h1 --backend ibm_cleveland --qubits 20 --submit   # send it
```

Nothing leaves the machine without `--submit`. Before submitting, the script transpiles every circuit at
`optimization_level=0` and asserts the delays survived: if the transpiler rescheduled or merged anything, it
aborts rather than running a schedule that is not the one that was designed.

`preflight.py` reads `dt`, the alignment granularity and the X-pulse duration, **solves the timing grid**, and
verifies that every schedule can be placed exactly. The grid step is not a formality. On an Eagle-class backend
`dt = 0.2222 ns` with a 16 dt alignment grid, a 16 µs window puts `T/8` at 9000 dt, which is not a multiple of
16, so the pulses could not sit where the schedule says. Preflight solves for the nearest legal window where all
five schedules place their pulse centres with zero error. A pulse that lands off-centre because a delay was
rounded is indistinguishable from the asymmetry the experiment deliberately introduces.

## The experiments

| | what it does | circuits | shots | ~QPU |
|---|---|---|---|---|
| `h1` | signed θ sweep × 4 schedules × 3 bases | 84 | 20k | ~9 min |
| `h2` | **N\* validation** — 2 hypotheses × 2 schedules × 3 bases, interleaved | 12 | 100k | ~6 min |
| `h3` | coherence vs window, no injected drift | 21 | 8k | ~1 min |

All qubits ride inside the same circuits, so per-qubit statistics cost nothing extra. `analyze.py` reports **per
qubit**, never pooled: the claim is that the verdict is a property of the schedule and not of one lucky qubit,
so the spread across qubits is itself a result.

The predictions were frozen before acquisition, in the `PREREGISTERED` table of `analyze.py`, and every
pre-registered outcome is reported afterwards, including the ones that do not clear their threshold.

## Why the drift injection is exact

A static Z-detuning of angular rate ω over a free interval τ is exactly `rz(ω·τ)`, and on IBM hardware `rz` is a
frame change: zero duration, no error. Because the injected drift is along σ_z and so is the dominant dephasing
generator, the two commute, so lumping a segment's continuous rotation into discrete frame updates leaves the
density matrix unchanged — an identity, not an approximation. Only amplitude damping fails to commute, which is
why each free segment is subdivided (default 8 pieces). The DD π-pulses then conjugate those frame rotations
with the correct signs automatically. The cancellation is not simulated; the device performs it.

Units: θ is an angular rate in **rad/µs**, matching `numerics/selfcheck_dd_idle_usecase.py`.
θ = 0.05 rad/µs = 5×10⁴ rad/s ≈ **7.96 kHz**.

**The θ sweep stays in the linear regime.** H1 fits a *first-order* slope and the unprotected signal is
`<Y> = -sin(θT)`. A sweep reaching θ = 0.10 rad/µs puts θT at 1.6 rad, where a straight-line fit recovers only
0.69 of the true slope. The sweep holds |θT| ≤ 0.16 rad (<0.5% nonlinearity) and `validate_local.py` fails the
run if it does not. A narrower sweep gives less slope precision, so XY4's suppression is reported as a lower
bound rather than a sharp number; the expansion parameter is θT, so it cannot be widened for the blind schedule
alone.

## What the local rehearsal cannot do

- **Aer cannot validate the protection claim.** Its noise model is Markovian, and dynamical decoupling suppresses
  *correlated* (1/f) noise, which the model does not contain. In simulation XY4 barely beats free evolution
  (~1.07×); on hardware it is 3.84×. `analyze.py` prints a warning when it sees the simulated case.
- **The bootstrap captures shot noise, not drift.** Resampling one acquired pool is exact for shot noise and
  blind to run-to-run drift, which is why the held-out estimator in `holdout_nstar.py` runs nine independent
  partitions and reports their median.

## Files

| file | role |
|---|---|
| `connect_check.py` | what an account can see: instances, allocation, backends |
| `pdet_hw.py` | library: timing grid, schedules, circuits, estimators, bootstrap N* |
| `preflight.py` | backend inspection, grid solve, qubit selection, budget |
| `validate_local.py` | ideal + noisy end-to-end validation |
| `run_experiment.py` | build, verify, dry-run / simulate / submit |
| `analyze.py` | per-qubit slopes, N*, coherence, against the pre-registered table |
| `holdout_nstar.py` | N* with the threshold learned from a disjoint calibration split |
| `make_paper_figures.py` | the device figures of the manuscript |
