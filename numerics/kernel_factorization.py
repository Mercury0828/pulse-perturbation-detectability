"""Separate the two reasons a coherent error is invisible, as a factorization.

The response factors through the toggling generator,

    Theta --K--> L --R_{S,O}--> R^{|S||O|},      M = R_{S,O} . K,

which gives two nested obstructions:

    ker K              control-blind: the schedule averages the error away, and
                       no measurement whatsoever can see it at first order.
    ker M / ker K      access-blind: the error survives the schedule but the
                       preparation/measurement budget cannot resolve it.

dim ker M = dim ker K + dim(ker M / ker K), so quoting either alone invites
confusion.

WHICH PULSE MODEL. This distinction only has content in the ideal-pulse limit,
because an exact rank is only defined there. An earlier version of this script
computed the ranks on a FINITE-WIDTH pulse schedule while the manuscript table
claimed ideal pulses, and the two disagreed with the paper's own proposition 1:
with 2.5% residuals a refocused echo direction is not in any numerical kernel,
and XY4's two surviving residuals happened to be linearly dependent, collapsing
the rank by accident. The table below is therefore computed with exact
instantaneous pi-pulses, where ker A is the group twirl of proposition 1, and
the finite-pulse numbers are reported separately as what they are: the
operational 5%-threshold criterion behind figure 2, not a rank.

Note also that the two figures use different dictionaries: figure 2 sweeps the
three single-qubit Pauli directions, the schedule x readout table sweeps the
transmon DRAG dictionary. Kernel dimensions are dictionary-relative and are not
comparable across the two.

Run: python kernel_factorization.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import models as M_
from pdet_core import Schedule, kernel_dim, response_map, toggling_generator

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "phase0")
os.makedirs(OUT, exist_ok=True)
RTOL = 1e-9

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)

SCHEDS = {"free": [], "echo": [(0.5, X)],
          "CPMG2": [(0.25, X), (0.75, X)],
          "XY4": [(0.125, X), (0.375, Y), (0.625, X), (0.875, Y)]}
ACCESS = {"Z-only": [Z], "comp+X": [Z, X], "full": [X, Y, Z]}


def dag(A):
    return A.conj().T


def st(v):
    v = np.array(v, complex)
    v = v / np.linalg.norm(v)
    return np.outer(v, v.conj())


STATES = [st([1, 0]), st([0, 1]), st([1, 1]), st([1, 1j])]


# ------------------------------------------------- exact ideal-pulse toggling


def ideal_toggling(pulses, Vlist):
    """K_j = int_0^T U_0(t)^dag V_j U_0(t) dt with instantaneous pi-pulses.

    Between pulses U_0 is constant, so this is an EXACT finite sum over the
    segments the pulses cut the window into -- no time discretisation, and in
    particular no off-by-one at a pulse boundary, which would leave a spurious
    residual of order 1/nseg exactly where the answer is zero. T is normalised
    to 1. Also returns U_0(T), needed to carry observables into the interaction
    frame."""
    fr = sorted(pulses)
    bounds = [0.0] + [f for f, _ in fr] + [1.0]
    Ks = [np.zeros((2, 2), complex) for _ in Vlist]
    U = I2.copy()
    for k in range(len(bounds) - 1):
        dt = bounds[k + 1] - bounds[k]
        if dt > 0:
            for i, V in enumerate(Vlist):
                Ks[i] += dt * (dag(U) @ V @ U)
        if k < len(fr):
            U = fr[k][1] @ U
    return Ks, U


def ideal_response(Ks, UT, states, obs):
    """M_{(l,s),j} = -i Tr[[rho_s, U_T^dag O_l U_T] K_j], as a real matrix."""
    rows = []
    for O in obs:
        Ot = dag(UT) @ O @ UT
        for r in states:
            c = r @ Ot - Ot @ r
            rows.append([float(np.real(-1j * np.trace(c @ K))) for K in Ks])
    return np.array(rows)


def rank_vs_scale(A, scale):
    """Numerical rank of A against an EXTERNAL scale.

    A relative tolerance referred to A's own largest singular value cannot
    detect that A is entirely zero, which is exactly the ideal-XY4 case: every
    toggling generator vanishes, so the whole matrix is zero and its kernel is
    everything. We therefore threshold against the un-decoupled generator norm."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    sv = np.linalg.svd(np.atleast_2d(A), compute_uv=False)
    return int(np.sum(sv > RTOL * scale))


def ideal_ker_K_dim(Ks, scale):
    G = np.array([K.reshape(-1) for K in Ks]).T
    return G.shape[1] - rank_vs_scale(np.vstack([G.real, G.imag]), scale)


# ------------------------------------------------------------------ reporting


def report_ideal(Vlist, names):
    print("\n" + "=" * 76)
    print("IDEAL PULSES -- exact ranks, the object proposition 1 describes")
    print("dictionary: single-qubit Pauli (%s)" % ", ".join(names))
    print("=" * 76)
    print("  %-8s %-9s %10s %14s %16s" % ("schedule", "access", "dim ker K", "dim ker M",
                                          "access-blind dim"))
    # the un-decoupled generator sets the scale every rank is measured against
    Kfree, UTfree = ideal_toggling([], Vlist)
    scale = max(float(np.linalg.norm(K)) for K in Kfree)
    mscale = max(1e-300, float(np.max(np.abs(ideal_response(Kfree, UTfree, STATES,
                                                            ACCESS["full"])))))
    rows = []
    for sn, pulses in SCHEDS.items():
        Ks, UT = ideal_toggling(pulses, Vlist)
        dk = ideal_ker_K_dim(Ks, scale)
        for an, O in ACCESS.items():
            Mmat = ideal_response(Ks, UT, STATES, O)
            dm = Mmat.shape[1] - rank_vs_scale(Mmat, mscale)
            assert dm >= dk, "ker K must be contained in ker M (%s/%s)" % (sn, an)
            rows.append({"schedule": sn, "access": an, "dim_ker_K": int(dk),
                         "dim_ker_M": int(dm), "access_blind_dim": int(dm - dk)})
            print("  %-8s %-9s %10d %14d %16d" % (sn, an, dk, dm, dm - dk))
    return rows


def report_ideal_norms(Vlist, names):
    """The per-direction toggling norms under ideal pulses, which is what makes
    the ranks above readable."""
    print("\n  toggling-generator norms under IDEAL pulses (relative to free):")
    print("  %-8s %14s %14s %14s" % ("schedule", "||K_X||", "||K_Y||", "||K_Z||"))
    base = None
    out = {}
    for sn, pulses in SCHEDS.items():
        Ks, _ = ideal_toggling(pulses, Vlist)
        row = [float(np.linalg.norm(K)) for K in Ks]
        if sn == "free":
            base = max(row)
        out[sn] = row
        print("  %-8s %14s %14s %14s"
              % (sn, *["%.2e (%.1f%%)" % (v, 100 * v / base) for v in row]))
    return {"baseline": base, "norms": out}


def sched_from_pulses(pulses):
    from phase4_evaluation import sched_from_pulses as _s
    return _s(pulses)


def report_finite_norms():
    """The finite-width-pulse norms behind figure 2's OPERATIONAL criterion.
    These are not ranks and must not be read as ker K."""
    print("\n" + "=" * 76)
    print("FINITE-WIDTH PULSES -- the operational 5% criterion behind figure 2")
    print("=" * 76)
    print("  Figure 2 calls a direction control-blind when ||K_V|| falls below 5% of")
    print("  the un-decoupled baseline. That is definition 2, not a rank: with finite")
    print("  pulses a refocused direction is small, not zero, so NO exact kernel exists")
    print("  and the ranks above are the ideal-pulse limit of these numbers.\n")
    print("  %-8s %14s %14s %14s %10s" % ("schedule", "||K_X||", "||K_Y||", "||K_Z||",
                                          "rank"))
    base = None
    norms = {}
    ranks = {}
    for sn, pulses in SCHEDS.items():
        sc = sched_from_pulses(pulses)
        Ks = [toggling_generator(sc, [V] * len(sc.H)) for V in (X, Y, Z)]
        row = [float(np.linalg.norm(K)) for K in Ks]
        if sn == "free":
            base = max(row)
        G = np.array([K.reshape(-1) for K in Ks]).T
        r = int(3 - kernel_dim(np.vstack([G.real, G.imag]), RTOL))
        norms[sn] = row
        ranks[sn] = r
        print("  %-8s %14s %14s %14s %10d"
              % (sn, *["%.2e (%.1f%%)" % (v, 100 * v / base) for v in row], r))
    print("\n  The numerical rank column is exactly why these must not be quoted as")
    print("  ker K: echo keeps 2.5% on two directions so its rank is full, and XY4's")
    print("  two 2.7% residuals happen to be linearly dependent so its rank collapses")
    print("  to 1. Neither number is the ideal-pulse answer.")
    return {"baseline": base, "norms": norms, "numerical_rank": ranks}


def main():
    out = {}
    pauli_names = ["X", "Y", "Z"]
    Vlist = [X, Y, Z]

    out["pauli_1q_ideal"] = report_ideal(Vlist, pauli_names)
    out["ideal_toggling_norms"] = report_ideal_norms(Vlist, pauli_names)
    out["finite_pulse_norms"] = report_finite_norms()

    # --- the transmon dictionary behind the schedule x readout table ----
    step_hams, dt = M_.transmon_x90_drag()[:2]
    Vdict = M_.dictionary_1q(len(step_hams), step_hams)
    names = list(Vdict.keys())
    print("\n" + "=" * 76)
    print("dictionary: transmon DRAG (the schedule x readout table)   (%d directions)"
          % len(names))
    print("=" * 76)
    print("  %s" % ", ".join(names))
    print("\n  Defined on the DRAG X90 gate, not on the idle schedules above. Kernel")
    print("  dimensions are dictionary-relative, so the two tables are not comparable")
    print("  and the manuscript says which is which.")
    out["transmon_dictionary"] = names

    with open(os.path.join(OUT, "kernel_factorization.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n  wrote results/phase0/kernel_factorization.json")

    print("\n" + "-" * 76)
    print("  Reading: dim ker M = dim ker K + access-blind dim. Enlarging the readout")
    print("  can only shrink the access-blind part; it never touches ker K.")
    return out


if __name__ == "__main__":
    main()
