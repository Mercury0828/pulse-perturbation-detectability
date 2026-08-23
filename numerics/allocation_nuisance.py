"""When does a single measurement setting carry the whole optimal budget?

The manuscript claimed, for a total shot budget split across separately measured
settings, that the whitened signal-to-noise ratio is linear in the allocation
weights w and therefore maximised at a vertex of the simplex -- i.e. spend
everything on one witness. That derivation treats the benign-projected signal h
as a constant. It is not: when the benign subspace B is non-trivial and its
coefficient is unknown, zeroing a setting's weight also removes that setting's
ability to pin the nuisance down, so the projection itself depends on w.

The allocation-dependent squared margin is

    gamma^2(w) = min_{b in B} sum_i (w_i / Sigma_ii) (d_i - (M b)_i)^2,

to be maximised over the simplex.  For each fixed b this is linear in w, so the
pointwise minimum is CONCAVE in w -- the optimum is generally interior, not a
vertex.  Only when B = {0} does the min disappear, gamma^2(w) becomes linear, and
the single-setting conclusion is recovered.

This script does three things:
  (a) reproduces the two-setting counterexample, where the naive rule says a
      single setting is optimal and the truth is that a single setting has zero
      margin at any shot count;
  (b) verifies concavity numerically and locates the true optimum;
  (c) checks which regime the paper's own reported costs live in.

Run: python allocation_nuisance.py -> ../results/phase0/allocation_nuisance.json
"""
from __future__ import annotations

import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "phase0")
os.makedirs(OUT, exist_ok=True)

SEED = 20260628


def margin2(w, d, MB, Sigma_diag, n_b_grid=20001, b_range=50.0):
    """gamma^2(w) = min_{b} sum_i (w_i/Sigma_ii) (d_i - (MB b)_i)^2.

    MB has the benign signal directions as columns. For a one-dimensional benign
    subspace the inner minimisation is solved on a dense grid, which is enough to
    exhibit the effect and avoids any claim resting on a closed form."""
    w = np.asarray(w, float)
    d = np.asarray(d, float)
    if MB.shape[1] == 0:
        return float(np.sum(w * d ** 2 / Sigma_diag))
    if MB.shape[1] != 1:
        raise NotImplementedError("grid minimiser written for a 1-D benign subspace")
    bs = np.linspace(-b_range, b_range, n_b_grid)
    resid = d[None, :] - bs[:, None] * MB[:, 0][None, :]          # (n_b, n_set)
    vals = np.sum((w / Sigma_diag)[None, :] * resid ** 2, axis=1)
    return float(np.min(vals))


def main():
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED}

    # ---------------------------------------------------------------- (a)
    d = np.array([1.0, -1.0])                 # target-error signal, differential
    MB = np.array([[1.0], [1.0]])             # benign drift, common mode
    Sig = np.array([1.0, 1.0])

    print("\n" + "=" * 78)
    print("(a) Two settings, unit variance. Benign = common mode span{(1,1)};")
    print("    target error = differential (1,-1), orthogonal to it.")
    print("=" * 78)

    naive_h = d - MB[:, 0] * float(d @ MB[:, 0]) / float(MB[:, 0] @ MB[:, 0])
    naive_per = naive_h ** 2 / Sig
    print("  benign-projected signal computed ONCE over both settings: h = %s" % naive_h)
    print("  the manuscript's rule reads off max_i h_i^2/Sigma_ii = %.3f and concludes"
          % naive_per.max())
    print("  that either single setting is optimal.\n")

    for name, w in (("all on setting 1", [1.0, 0.0]),
                    ("all on setting 2", [0.0, 1.0]),
                    ("split 50/50", [0.5, 0.5])):
        g2 = margin2(w, d, MB, Sig)
        print("  w = %-18s gamma^2(w) = %.6f" % (str(w), g2))
    print("\n  A single setting has gamma^2 = 0: the benign coefficient is unknown and")
    print("  unbounded, so on one coordinate the benign and error families of")
    print("  distributions coincide exactly and NO shot count separates them. The naive")
    print("  rule reports a finite cost where the true cost is infinite.")

    res["counterexample"] = {
        "d": d.tolist(), "benign_span": MB[:, 0].tolist(),
        "naive_projected_h": naive_h.tolist(),
        "naive_max_per_setting_ratio": float(naive_per.max()),
        "gamma2_vertex_1": margin2([1.0, 0.0], d, MB, Sig),
        "gamma2_vertex_2": margin2([0.0, 1.0], d, MB, Sig),
        "gamma2_even_split": margin2([0.5, 0.5], d, MB, Sig),
    }

    # ---------------------------------------------------------------- (b)
    print("\n" + "=" * 78)
    print("(b) gamma^2(w) is concave, so the optimum is interior, not a vertex")
    print("=" * 78)
    ts = np.linspace(0.0, 1.0, 21)
    curve = [(float(t), margin2([t, 1 - t], d, MB, Sig)) for t in ts]
    print("  %-8s %s" % ("w_1", "gamma^2"))
    for t, g in curve[::2]:
        print("  %-8.2f %.6f" % (t, g))
    best_t, best_g = max(curve, key=lambda p: p[1])
    print("\n  optimum at w_1 = %.2f, gamma^2 = %.6f -- an interior point." % (best_t, best_g))

    # concavity check on random chords
    bad = 0
    for _ in range(2000):
        t1, t2 = rng.random(), rng.random()
        lam = rng.random()
        tm = lam * t1 + (1 - lam) * t2
        g1 = margin2([t1, 1 - t1], d, MB, Sig)
        g2_ = margin2([t2, 1 - t2], d, MB, Sig)
        gm = margin2([tm, 1 - tm], d, MB, Sig)
        if gm < lam * g1 + (1 - lam) * g2_ - 1e-9:
            bad += 1
    print("  concavity violated on %d of 2000 random chords (expected 0): min of a" % bad)
    print("  family of functions linear in w is concave in w.")
    res["concavity"] = {"curve": curve, "argmax_w1": best_t, "max_gamma2": best_g,
                        "chord_violations": bad}

    # ---------------------------------------------------------------- (c)
    print("\n" + "=" * 78)
    print("(c) With B = {0} the vertex conclusion is recovered")
    print("=" * 78)
    MB0 = np.zeros((2, 0))
    curve0 = [(float(t), margin2([t, 1 - t], d, MB0, Sig)) for t in ts]
    b0 = max(curve0, key=lambda p: p[1])
    print("  gamma^2(w) is now linear in w; optimum at w_1 = %.2f, gamma^2 = %.3f,"
          % (b0[0], b0[1]))
    print("  attained at a vertex, which is the manuscript's original claim.")
    print("\n  This is the regime the paper's own reported costs live in: both the frozen")
    print("  prediction and the hardware analysis compare theta = 0 against a fixed theta")
    print("  on the same setting, with no benign subspace projected out, so B = {0} and")
    print("  the single-witness convention is optimal there. The proposition now says so")
    print("  instead of claiming it in general.")
    res["B_is_zero"] = {"curve": curve0, "argmax_w1": b0[0], "max_gamma2": b0[1],
                        "vertex_optimal": bool(abs(b0[0] - round(b0[0])) < 1e-9)}

    res["conclusion"] = (
        "The single-setting allocation optimum holds iff the inner minimisation over the "
        "benign set is vacuous (B = {0}, or a known benign point). For a non-trivial "
        "unknown benign subspace gamma^2(w) is concave rather than linear, the optimum is "
        "generally interior, and a vertex allocation can have zero margin where the full "
        "allocation has a positive one.")

    with open(os.path.join(OUT, "allocation_nuisance.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("\n  wrote results/phase0/allocation_nuisance.json")
    return res


if __name__ == "__main__":
    main()
