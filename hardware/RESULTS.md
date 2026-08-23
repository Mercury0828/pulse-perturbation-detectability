# PDET on `ibm_cleveland` — measured results

First hardware validation of the PDET workflow, run 2026-08-20 on the Cleveland Clinic Quantum Computer
(`ibm_cleveland`, Heron r2, 156 qubits) through the `DQC methods` on-prem instance.

**Every prediction in the "pre-registered" columns was computed from `numerics/` before the device was
touched**, and is the number the manuscript already quotes from
`results/selfcheck/dd_idle_usecase_results.json`. Nothing was tuned after seeing the data.

All experiments ran on **20 qubits simultaneously** (pairwise ≥2 hops apart), so every figure below is a
distribution over 20 independent replicas rather than one qubit's story.

| | |
|---|---|
| Window | 4000 dt = **16.0000 µs** exactly (solved onto the backend grid; every π-pulse centre placed with zero error) |
| Injected drift | θ = 0.05 rad/µs ≈ **7.96 kHz**, applied as virtual-Z frame updates during the free segments |
| Qubits | 9, 16, 24, 27, 34, 39, 44, 47, 51, 54, 59, 68, 79, 81, 96, 130, 137, 140, 144, 151 |
| Readout correction | p = 0.006 |

---

## H2 — detection cost N\* (the flagship)

Job `da3nk5m1vhnc73fk2ufg`, 12 circuits × 100,000 shots, **338 s** of allocation.

**PDET predicted 9,310 shots. The device delivered a median of 8,457 across 20 qubits — agreement to 9%.**

| | pre-registered | measured |
|---|---|---|
| N\* on the exposing edit (`xy4_asym`) | 9,310 shots | **8,457** (median, analytic) |
| observed / predicted | band [0.5, 2] | **0.91** (analytic), 1.18 (empirical, bootstrap) |
| qubits inside [0.3, 3] | ≥ 80% | **100% (20/20)** |
| witness setting | ⟨Y⟩ | **⟨Y⟩ on 20/20 qubits** |

Per-qubit N\* ranged 5,730–15,756 (IQR 6,989–13,000); γ ranged 0.0265–0.0440.

**The control is as important as the result.** Under the unmodified production schedule (`xy4`) the same
drift costs a median of **495,996 shots** — **59× more** — and on 14 of the 20 qubits the bootstrap found no
crossing at all below 3×10⁵ shots. The blind spot is expensive on hardware, exactly as the workflow says,
and the prescribed schedule edit is what makes it affordable.

Both figures are read on the witness the kernel prescribes, ⟨Y⟩. Selecting the witness by the largest
observed separation instead would bias the blind schedule's cost downwards, because on a schedule whose raw
response sits below the pure-noise norm the maximum of three noisy separations is drawn away from zero. On the
exposing edit the prescribed witness is also the empirical maximum on 20 of 20 qubits, so the two agree wherever
the response is resolved.

## H1 — first-order response and the blind spot

Job `da3nrvbotlns7399ufk0`, 84 circuits × 20,000 shots, **472 s** of allocation.
Signed θ sweep over ±0.01 rad/µs (|θT| ≤ 0.16 rad, <0.5% nonlinearity).

| schedule | measured ‖ds/dθ‖ (debiased median) | pre-registered | ratio |
|---|---|---|---|
| free | 14.302 | 14.00 | **1.022** |
| XY4 drop-1 | 3.760 | 3.33 | 1.129 |
| **asym. XY4** | **0.696** | **0.700** | **0.995** |
| XY4 | 0.107 | 2×10⁻⁵ (ideal-pulse limit) | consistent with zero |

**XY4's first-order signal is not resolvable.** Its raw response norm (0.562) sits *below* the norm a
pure-noise vector would produce at this shot count (0.624), so the correct statement is a bound:
**suppression > 33× at 2σ**, not a measured value. The 2σ upper limit on the median, 0.427, is the
97.5th percentile of a bootstrap of the median over the 20 qubits; the debiased per-qubit responses are
bimodal (ten are exactly zero after debiasing), so scaling the per-component slope error would be wrong. That bound — not the simulation's 2×10⁻⁵, which is the
ideal-pulse limit — is the honest number for the manuscript's finite-pulse remark.

Two analysis points worth carrying into the paper:

- The response *norm* is positively biased (three noisy components give E[norm²] = |true|² + 3σ²); it is
  debiased above. Without that correction XY4 appears to have a 26× "measured suppression" that is entirely
  noise.
- Projecting on ⟨Y⟩ instead is **not** a valid fix. On an unprotected schedule the qubit's own static
  detuning accumulates over 16 µs and rotates the response out of Y, so the projection under-reports free
  evolution by ~30%. Decoupled schedules refocus that static detuning, so their response does stay in Y.
  The debiased norm is right for both.

## H3 — protection, measured rather than assumed

Job `da3o15jotlns7399ulvg`, 21 circuits × 20,000 shots, **128 s** of allocation.
(A 1,000-shot probe, job `da3mqn6aa69c739irie0`, 9 s, was run first to calibrate the cost model.)

| schedule | measured T₂^seq | vs free |
|---|---|---|
| free | 80.7 µs | 1.00 |
| **XY4** | **310.1 µs** | **3.84×** |
| asym. XY4 | 259.7 µs | 3.22× |

**Retention (asym / XY4) = 0.877**, 95% CI [0.782, 0.939] over 20 qubits.

The pre-registered threshold was ≥ 0.85. It lies **inside** the confidence interval, so the prediction is
**consistent but not sharply resolved**: the per-qubit spread is wide (IQR 0.754–0.941, full range
0.267–1.034) and only 60% of qubits individually clear 0.85. The pre-registered ordering
XY4 > asym > free holds on 80% of qubits. Reporting this as a clean pass would overstate it; the honest
statement is that the trade-off survives contact with hardware at the ~15% level, with real qubit-to-qubit
variation that the simulation does not capture.

This is the one claim local simulation *could not* test: Aer's noise model is Markovian, decoupling
suppresses correlated (1/f) noise, and in simulation XY4 beat free evolution by only 1.07×. On hardware it
is 3.84×. The physics is self-consistent — free evolution from |+⟩ measures T₂\* (80.7 µs, no refocusing),
and XY4 recovers 310 µs, approaching the backend's reported Hahn T₂ ≈ 380 µs.

The manuscript's 0.93 retention is a filter-function ratio over an **assumed** 1/f spectrum; 0.877 is the
**measured** coherence ratio and is the number to quote.

### An analysis trap worth recording

The first pass computed `median(T2_asym) / median(T2_xy4)` and got 0.838, which would have been reported as
a *failed* pre-registered prediction. The right statistic is the **median of the per-qubit ratios**, 0.877,
which is consistent with the threshold. A ratio of medians is not the median of ratios, and here the
difference straddled the decision boundary. `analyze.py` now computes the per-qubit form with a bootstrap CI.

## Cost model, calibrated

Per shot = window + 2.652 µs readout + **260.9 µs** overhead (within 5% of the backend's
`default_rep_delay` of 250 µs).

| job | predicted | charged | error |
|---|---|---|---|
| H2 | 335 s | 338 s | **0.9%** |
| H1 | 470 s | 472 s | **0.4%** |

The calibration probe charged 9 s against 6.30 s of circuit execution (1.43×); that markup is a per-job
load overhead and amortises away on shot-heavy jobs, as the two rows above show.

**Allocation used: 947 s of 1800 s** (H2 338 s, H1 472 s, H3 128 s, probe 9 s).

## Reproducing

```bash
cd hardware
python preflight.py --backend ibm_cleveland          # zero QPU cost
python validate_local.py --backend ibm_cleveland     # zero QPU cost
python analyze.py "results_hw/h2_ibm_cleveland_*.json" --readout-error 0.006
```

Raw counts, the resolved timing grid, the qubit set, the git commit and the charged time are all in the
`results_hw/*.json` provenance blocks.
