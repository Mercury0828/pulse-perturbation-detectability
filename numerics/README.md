# PDET numerics — reproducibility

All results in the paper regenerate from this directory. **Fixed seed 20260628** in every module; ~minutes total.
Numerics are validated against exact propagation by the **A1 correctness gate** (`pdet_core.py` self-test):
the closed-form response agrees with the exact finite-difference derivative of the propagated expectations to a
**maximum relative error of 7e-8** across every model in the suite, and **1.4e-10** on the two-level
reference case, well inside the gate's `< 1e-4` pass requirement. `run_all.py` exits nonzero if any module fails.

## Regenerate everything
```
pip install -r requirements.txt
python run_all.py          # runs every module in dependency order; A1 gate first
```
Or run any single module (each writes its own artifacts under `../results/`):
```
python pdet_core.py                      # A1 correctness gate (must pass first)
python phase4_evaluation.py              # Table 1 + Fig 1-3 (MC + 95% CI)
python selfcheck_dd_idle_usecase.py      # the non-contrived flagship (DD on idle qubit)
...
```

## Module map
| module | what it produces | results/ |
|---|---|---|
| `pdet_core.py` | core response map `M`, `ker M`, `gamma`, `eta2` + **A1 gate** | (self-test) |
| `models.py` | transmon DRAG + cross-resonance + leakage models, schedules, dictionaries | — |
| `baselines.py` | GST-lite / shadows / Ham-learning / channel-cert / QFI / Jacobian / rubric | — |
| `run_phase0.py` | de-risk: kernel/γ/η₂, baseline bake-off, Go/No-go | `phase0/` |
| `phase1b_dd_blindspot.py` | genuine control knob (DD blind spot) | `phase1/` |
| `phase1c/d_*.py` | filter-function + Fisher baselines | `phase1/` |
| `a3_upper_bound_check.py`, `a4_kerA_characterization.py` | A3 bound (tight) + A4 `ker A` | (console) |
| `phase4_evaluation.py` | Table 1 + Fig 1–3 (MC-validated, 95% CI) | `phase4/` |
| `phase4b/c/d_*.py` | FWER discovery, equal-budget, K_eff, SPAM, 1/f protection, nuisance | `phase4/` |
| `selfcheck_realistic_noise.py` | **L1** open-system T1/T2/readout (qutip Lindblad) | `selfcheck/` |
| `selfcheck_dd_idle_usecase.py` | **non-contrived flagship**: DD-on-idle blind spot + diagnostic | `selfcheck/` |
| `selfcheck_baseline_headtohead.py` | equal-budget baselines (clean win vs measurement-only) | `selfcheck/` |
| `selfcheck_scalability.py` | **L6** locality-scaled O(n) diagnosis + spreading boundary | `selfcheck/` |

## Environment
Python 3.11.4; pins in `requirements.txt` (numpy 1.26.4, scipy 1.16.1, qutip 5.1.1, matplotlib 3.8.4,
stim 1.16.0), verified by importing each package and reading `__version__`. Open-system noise uses QuTiP
Lindblad; Clifford operator-spreading uses stim. The `hardware/` scripts carry their own qiskit
dependencies and are not needed to reproduce any simulation result.

## Data / Code availability statement (for the paper)
All simulation code and the scripts that regenerate every figure/table are provided in this repository under an
open-source license; each figure is reproduced by a single named script with a fixed seed. No proprietary data or
hardware access is required for anything in this directory. The device results of the manuscript live in
`../hardware/`, which carries the acquired records, the analysis, and its own README.
