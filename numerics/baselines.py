"""
PDET Phase-0 baseline battery (functional minimal, 1-2 qubit) for the GST+shadows EQUIVALENCE RUBRIC (the RED
test, phase0_spec.md §5/§6). Implements the practitioner pipeline PDET must beat (GST+shadows equivalence rubric):
  - Jacobian-nullspace  (the "too-thin" baseline: d<O>/dtheta nullspace == PDET visibility map?)
  - GST / error-generator identifiability (rank of the same experiment-class Jacobian; gauge note)
  - Hamiltonian-learning / system-ID (LS recovery; which directions are unrecoverable)
  - control-optimized QFI/CFI (single-direction first-order Fisher)
  - channel certification (worst-case, access-blind reference)
Each returns what it OUTPUTS and whether a practitioner can read off (i) per-direction visibility verdict and
(ii) an actionable control-schedule change. The rubric() function renders the PASS/RED decision.
"""
from __future__ import annotations
import numpy as np
from pdet_core import (Schedule, toggling_generator, response_map, kernel_dim, kernel_basis, singular_spectrum,
                       perturbed_signal)

RTOL = 1e-9

# ----------------------------------------------------------------------------- (1) Jacobian-nullspace baseline
def jacobian_nullspace(a1_builder, Vlists, S, O, eps=1e-6):
    """
    Off-the-shelf practitioner move: numerically estimate the Jacobian J_{(l,s),j} = d<O_l>_{rho_s}/d theta_j by
    finite differences (NO PDET internals), then take its nullspace. Returns (J, dim_null, J).
    This is INDEPENDENT of PDET's analytic M -- used to test whether PDET's visibility map is 'beyond baseline'.
    """
    m = len(Vlists)
    cols = []
    for j in range(m):
        e = np.zeros(m); e[j] = 1.0
        gp = perturbed_signal(a1_builder, eps * e, Vlists, S, O)
        gm = perturbed_signal(a1_builder, -eps * e, Vlists, S, O)
        cols.append(((gp - gm) / (2 * eps)).reshape(-1))
    J = np.stack(cols, axis=1)   # (L*S, m)
    s = singular_spectrum(J)
    rank = int(np.sum(s > RTOL * s[0])) if s.size else 0
    return J, m - rank

# ----------------------------------------------------------------------------- (2) GST / error-generator rank
def gst_identifiability(M):
    """
    GST estimates error-generator (Lindbladian) coefficients on the same experiment class. On a single fixed
    schedule with access (S,O), the first-order identifiable subspace is exactly row-space(M); the unidentifiable
    set is ker M (plus gauge). Returns identifiable_rank, unidentifiable_dim.
    """
    s = singular_spectrum(M)
    rank = int(np.sum(s > RTOL * s[0])) if s.size else 0
    return rank, M.shape[1] - rank

# ----------------------------------------------------------------------------- (3) Hamiltonian-learning (LS)
def hamiltonian_learning(M, seed=0, noise=0.0):
    """
    Least-squares recovery of theta from first-order data y = M theta (+ optional noise). Directions in ker M are
    unrecoverable. Returns per-direction recovery error for unit perturbations along each axis.
    """
    rng = np.random.default_rng(seed)
    m = M.shape[1]
    Mpinv = np.linalg.pinv(M)
    errs = {}
    for j in range(m):
        e = np.zeros(m); e[j] = 1.0
        y = M @ e
        if noise: y = y + noise * rng.standard_normal(y.shape)
        est = Mpinv @ y
        errs[j] = float(np.linalg.norm(est - e))
    return errs

# ----------------------------------------------------------------------------- (4) control-optimized QFI/CFI
def qfi_single_direction(M):
    """First-order CFI per direction = ||M e_j||^2 (single-direction sensitivity). Returns dict j->CFI."""
    return {j: float(np.sum(M[:, j] ** 2)) for j in range(M.shape[1])}

# ----------------------------------------------------------------------------- (5) channel certification (worst-case)
def channel_certification_refnote():
    """Access-blind worst-case reference: diamond-distance certification bounds the WHOLE channel over ALL
    states/observables. It does not resolve which restricted-access perturbation DIRECTION is invisible, and gives
    a worst-case (often loose) bound. Conceptual baseline -- not run numerically in Phase 0."""
    return {"output": "worst-case whole-channel distinguishability bound (access-blind)",
            "per_direction_visibility": False, "actionable_control_change": False}

# ----------------------------------------------------------------------------- equivalence rubric (RED test)
def rubric(pdet_dim_ker, jac_dim_null, knob_passes_anywhere):
    """
    Renders the frozen GST+shadows equivalence rubric.
    (i)  per-direction visibility verdict: is PDET's ker M reproduced by the off-the-shelf Jacobian nullspace?
         If YES, the visibility MAP is NOT 'beyond baseline'.
    (ii) actionable control-schedule change: does any baseline directly output it? (No: GST/shadows/Ham-learning/
         QFI/cert all characterize a FIXED schedule; none prescribes changing U0(t) to shrink ker M.)
    PASS (engineering delta real) iff PDET delivers (ii) -- an actionable control change -- that baselines do not,
    even if (i) is reproducible. RED iff PDET delivers NOTHING beyond baseline (neither (i) nor (ii)).
    """
    map_reproducible = (jac_dim_null == pdet_dim_ker)
    baseline_gives_control_change = False  # established by construction: no listed baseline changes U0(t)
    pdet_gives_control_change = bool(knob_passes_anywhere)
    delta_real = pdet_gives_control_change and not baseline_gives_control_change
    verdict = "PASS" if delta_real else ("RED" if not pdet_gives_control_change else "PASS")
    return {
        "visibility_map_reproducible_by_jacobian": bool(map_reproducible),
        "visibility_map_beyond_baseline": bool(not map_reproducible),
        "baseline_produces_actionable_control_change": baseline_gives_control_change,
        "pdet_produces_actionable_control_change": pdet_gives_control_change,
        "engineering_delta_is_the_control_knob_plus_finite_shot": bool(delta_real),
        "rubric_verdict": verdict,
        "note": ("PDET's visibility MAP is the observable-Jacobian nullspace (NOT novel -- a known tool, honesty "
                 "anchor N0). The delta that survives the rubric is the CONTROL-DESIGN KNOB (shrink ker M / raise "
                 "gamma by changing U0(t)) + honest finite-shot, which no listed baseline directly outputs.")
    }
