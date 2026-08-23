"""Two constraints the workflow has to carry, and a third verdict it has to return.

A referee pointed out that the schedule-edit search of proposition 1 is stated
without the constraint that makes it usable -- an edit is only admissible if the
edited schedule still implements the gate it replaced -- and that the
readout-blind branch of algorithm 1 conflates two cases with different fixes.
This script computes both, so the manuscript can state them as checked facts.

Part A: target preservation.  For an idle block the target is the identity.  We
compute the net ideal-pulse word of every candidate edit, and the actual
finite-pulse propagator U_0'(T), and report the target infidelity of each.  The
result decides which edits are admissible without a compensating frame update.

Part B: preparation-blind vs measurement-blind.  When an error survives the
schedule (L_theta != 0) but M theta = 0, there are two cases.  If some available
state fails to commute with L_theta, a basis change exposes the error and the
workflow returns an observable.  If every available state commutes with it, no
observable whatsoever helps and the fix is a PREPARATION edit.  We compute
max_s ||[rho_s, L_theta]|| for a ladder of preparation budgets and show the
verdict flipping.

Run: python algorithm_constraints.py -> ../results/phase0/algorithm_constraints.json
"""
from __future__ import annotations

import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "phase0")
os.makedirs(OUT, exist_ok=True)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}

# fractional pulse time and axis, matching selfcheck_dd_idle_usecase
SCHEDULES = {
    "free":      [],
    "echo":      [(0.500, "X")],
    "CPMG2":     [(0.250, "X"), (0.750, "X")],
    "XY4":       [(0.125, "X"), (0.375, "Y"), (0.625, "X"), (0.875, "Y")],
    "XY4 drop-1": [(0.125, "X"), (0.375, "Y"), (0.625, "X")],
    "XY4 asym":  [(0.100, "X"), (0.375, "Y"), (0.625, "X"), (0.875, "Y")],
}
EDIT_KIND = {"free": "baseline", "echo": "baseline", "CPMG2": "baseline",
             "XY4": "production", "XY4 drop-1": "pulse removal", "XY4 asym": "timing only"}

TOL = 1e-9


def dag(A):
    return A.conj().T


# ------------------------------------------------------- A: target preservation


def pulse_word(pulses):
    """Net ideal-pulse unitary, later pulses applied on the left."""
    W = I2.copy()
    for _, ax in sorted(pulses):
        W = PAULI[ax] @ W
    return W


def finite_pulse_propagator(pulses, nfree=160, pulse_frac=1.0 / 50.0):
    """U_0'(T) with the same finite-width pulse model as the open-system study:
    each pi-pulse is a square drive of duration dt/50 inside its step."""
    dt = 1.0 / nfree
    steps = {int(round(f * nfree)): ax for f, ax in pulses}
    U = I2.copy()
    for k in range(nfree):
        # idle: no drift, so the free step is the identity
        if k in steps:
            pi_dt = dt * pulse_frac
            H = (np.pi / pi_dt / 2.0) * PAULI[steps[k]]
            w, v = np.linalg.eigh(H)
            U = (v @ np.diag(np.exp(-1j * w * pi_dt)) @ dag(v)) @ U
    return U


def phase_infidelity(U, target=None):
    """1 - |Tr(target^dag U)|/d : zero iff U equals the target up to a phase."""
    target = I2 if target is None else target
    return float(1.0 - abs(np.trace(dag(target) @ U)) / U.shape[0])


def nearest_pauli(U):
    """If U is proportional to a Pauli, name it: such a residual can be tracked in
    the software Pauli frame at no cost for a Clifford continuation."""
    best, bi = None, 1.0
    for name, P in PAULI.items():
        inf = phase_infidelity(U, P)
        if inf < bi:
            best, bi = name, inf
    return (best, bi) if bi < TOL else (None, bi)


def part_a():
    print("\n" + "=" * 84)
    print("A. Does the edited schedule still implement the block it replaced?")
    print("=" * 84)
    print("  Target for an idle block is U_target = I (up to a global phase).")
    print("\n  %-12s %-14s %-10s %14s %14s %s"
          % ("schedule", "edit kind", "word", "infid (ideal)", "infid (finite)", "residual"))
    rows = {}
    for name, pulses in SCHEDULES.items():
        W = pulse_word(pulses)
        Uf = finite_pulse_propagator(pulses)
        inf_i = phase_infidelity(W)
        inf_f = phase_infidelity(Uf)
        pname, _ = nearest_pauli(W)
        word = "".join(ax for _, ax in sorted(pulses)) or "-"
        ok = inf_i < TOL
        note = "identity" if ok else ("Pauli %s (frame-trackable)" % pname if pname else "non-Pauli")
        rows[name] = {"edit_kind": EDIT_KIND[name], "pulse_word": word,
                      "target_infidelity_ideal": round(inf_i, 12),
                      "target_infidelity_finite_pulse": round(inf_f, 12),
                      "preserves_target": bool(ok),
                      "residual": note}
        print("  %-12s %-14s %-10s %14.2e %14.2e %s"
              % (name, EDIT_KIND[name], word, inf_i, inf_f, note))
    print("\n  Reading. The prescribed edit is a TIMING change: it moves a pulse without")
    print("  adding or removing one, so the pulse word is unchanged and target")
    print("  preservation is automatic -- exactly, not approximately.  Dropping a pulse")
    print("  changes the word and leaves a residual Pauli, so it is admissible only if a")
    print("  compensating frame update is applied; that is free in software for a Clifford")
    print("  continuation and not free otherwise.  The manuscript reports both, but only")
    print("  the timing edit is prescribed by the workflow.")
    return rows


# ----------------------------------------- B: preparation- vs measurement-blind


def toggling_generators(pulses, Vlist):
    """K_j = int_0^T U_0(t)^dag V_j U_0(t) dt for an idle schedule with
    instantaneous pi-pulses.

    U_0 is constant between pulses, so this is an EXACT finite sum over the
    segments the pulses cut the window into. A time-stepped version instead puts
    an off-by-one segment on each side of every pulse, which leaves a residual of
    order 1/nsteps precisely where the answer is zero -- enough to misclassify a
    symmetric echo as access-blind when it is control-blind. T is normalised to 1."""
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
            U = PAULI[fr[k][1]] @ U
    return Ks, U


def st(v):
    v = np.array(v, complex)
    v = v / np.linalg.norm(v)
    return np.outer(v, v.conj())


PREP_SETS = {
    "|0> only":        [st([1, 0])],
    "|0>,|1>":         [st([1, 0]), st([0, 1])],
    "|0>,|+>":         [st([1, 0]), st([1, 1])],
    "|0>,|+>,|+i>":    [st([1, 0]), st([1, 1]), st([1, 1j])],
}
OBS_SETS = {"Z-only": ["Z"], "full": ["X", "Y", "Z"]}


def classify(pulses, states, obs_names, theta, Vlist, rtol=1e-9):
    """Return the verdict for one error direction under one access budget.

    Thresholds are relative to the same direction under free evolution, so that
    'zero' means zero on the scale of the un-decoupled generator rather than on
    an arbitrary absolute scale."""
    Kfree, UTfree = toggling_generators([], Vlist)
    scale = float(np.linalg.norm(sum(t * K for t, K in zip(theta, Kfree))))
    eps = rtol * max(scale, 1e-300)
    Ks, UT = toggling_generators(pulses, Vlist)
    L = sum(t * K for t, K in zip(theta, Ks))
    if np.linalg.norm(L) <= eps:
        return "control-blind", 0.0, None
    Otil = {n: dag(UT) @ PAULI[n] @ UT for n in obs_names}
    sig = max(abs(np.trace(-1j * (r @ o - o @ r) @ L))
              for r in states for o in Otil.values())
    if sig > eps:
        return "visible", float(sig), None
    # M theta = 0 with L != 0: which edit fixes it?
    comm = max(float(np.linalg.norm(r @ L - L @ r)) for r in states)
    if comm <= eps:
        return "preparation-blind", 0.0, None
    # some available state does move; find the observable that reads it out
    r = max(states, key=lambda r: np.linalg.norm(r @ L - L @ r))
    W = 1j * (r @ L - L @ r)                      # the ideal witness in the interaction frame
    Wlab = UT @ W @ dag(UT)                       # carried back to the lab frame
    best = max(PAULI, key=lambda n: abs(np.trace(dag(PAULI[n]) @ Wlab)))
    return "measurement-blind", 0.0, best


def part_b():
    print("\n" + "=" * 84)
    print("B. When no available observable works, the fix is a preparation edit")
    print("=" * 84)
    print("  Error direction: a static Z-detuning.  Dictionary {X, Y, Z}, ideal pulses.")
    print("\n  %-10s %-16s %-8s %-20s %s"
          % ("schedule", "preparations", "readout", "verdict", "prescribed fix"))
    rows = []
    Vlist = [X, Y, Z]
    theta = [0.0, 0.0, 1.0]                       # the Z-detuning direction
    for sname in ("free", "echo", "XY4", "XY4 asym"):
        pulses = SCHEDULES[sname]
        for pname, states in PREP_SETS.items():
            for oname, obs in OBS_SETS.items():
                verdict, sig, wit = classify(pulses, states, obs, theta, Vlist)
                fix = {"visible": "-",
                       "control-blind": "schedule edit",
                       "measurement-blind": "measure %s" % wit,
                       "preparation-blind": "prepare a state off the Z axis"}[verdict]
                rows.append({"schedule": sname, "preparations": pname, "readout": oname,
                             "verdict": verdict, "signal": round(sig, 9), "witness": wit})
                print("  %-10s %-16s %-8s %-20s %s" % (sname, pname, oname, verdict, fix))
        print()
    print("  Reading. With only |0> preparable, a Z-detuning is invisible in EVERY basis")
    print("  even under free evolution -- [rho, L] = 0, so no observable has a first-order")
    print("  response.  Enlarging the readout cannot fix it and the workflow must say so;")
    print("  adding |+> to the preparation set makes the same error measurement-blind, and")
    print("  then a Y measurement exposes it.  The two cases are distinguished by one test,")
    print("  max_s ||[rho_s, L_theta]||, which is what algorithm 1 now runs.")
    return rows


def main():
    res = {"target_preservation": part_a(), "prep_vs_measurement": part_b()}
    with open(os.path.join(OUT, "algorithm_constraints.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("\n  wrote results/phase0/algorithm_constraints.json")
    return res


if __name__ == "__main__":
    main()
