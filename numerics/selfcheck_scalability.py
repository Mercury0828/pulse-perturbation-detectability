"""
SELF-CHECK L6 -- scalability / path-to-scale (hostile-review must-do #5; honest locality-scaled claim).

Honest thesis (matches the reviewer's prediction): the K-level detectability KERNEL for LOCAL coherent
perturbations under per-qubit dynamical decoupling is **locality-scaled** -- it FACTORIZES over the qubits each
perturbation touches, so computing it costs O(dictionary size) = O(n), INDEPENDENT of the full 2^n Hilbert space.
The boundary is ENTANGLING control: it spreads the toggling-frame generator to high-weight Pauli strings (stim),
after which the local-dictionary picture breaks. Claim = locality-scaled diagnosis, NOT general scalable tomography.

We demonstrate for n=2..7:
  (1) build a LOCAL dictionary (single-qubit {X,Y,Z} on each qubit + nearest-neighbor {ZZ,XX}); size O(n).
  (2) under per-qubit XY4, classify each local perturbation K-blind / visible via a 1-2 qubit toggling computation
      (cost independent of n) -> O(n) total; report kernel dim + runtime + memory (dict only).
  (3) operator-spreading boundary: a single-qubit Pauli evolved through random entangling Clifford layers (stim)
      -> mean weight grows with depth -> the locality assumption fails under entangling control (honest limit).

Run: python selfcheck_scalability.py -> ../results/selfcheck/scalability_results.json + fig.
"""
from __future__ import annotations
import json, os, time
import numpy as np
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "selfcheck"); os.makedirs(OUT, exist_ok=True)
SEED = 20260628
TIMING_REPEATS = 9   # wall-clock timings are noisy; the reported figure is the median of repeats
I = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]],complex); Y = np.array([[0,-1j],[1j,0]],complex); Z = np.array([[1,0],[0,-1]],complex)

# ---------------------------------------------------------------------- the actual toggling computation
PAULI = {"I": I, "X": X, "Y": Y, "Z": Z}
XY4_PULSES = [(0.125, "X"), (0.375, "Y"), (0.625, "X"), (0.875, "Y")]   # per-qubit XY4, synchronous on the support

def _string_op(word, nq):
    """The Pauli string `word` (one letter per qubit of the support) as a 2^nq matrix."""
    out = np.array([[1.0 + 0j]])
    for ch in word:
        out = np.kron(out, PAULI[ch])
    assert out.shape == (2 ** nq, 2 ** nq)
    return out

def _xy4_schedule(nq):
    """Ideal instantaneous per-qubit XY4 applied simultaneously to every qubit of the support."""
    return [(f, _string_op(ax * nq, nq)) for f, ax in XY4_PULSES]

def toggling_average(pulses, V):
    """K = int_0^1 U0(t)^dag V U0(t) dt for ideal instantaneous pulses.

    Between pulses U0 is constant, so this is an EXACT finite sum over the segments the pulses cut the
    window into: no time discretisation, and in particular no off-by-one at a pulse boundary, which would
    leave a spurious residual of order 1/nseg exactly where the answer is zero."""
    fr = sorted(pulses)
    bounds = [0.0] + [f for f, _ in fr] + [1.0]
    K = np.zeros_like(V)
    U = np.eye(V.shape[0], dtype=complex)
    for k in range(len(bounds) - 1):
        dt = bounds[k + 1] - bounds[k]
        if dt > 0:
            K += dt * (U.conj().T @ V @ U)
        if k < len(fr):
            U = fr[k][1] @ U
    return K

def local_dictionary(n):
    """O(n) local dictionary: single-qubit X,Y,Z on each qubit + nearest-neighbor ZZ, XX."""
    d = []
    for q in range(n):
        for p in ["X", "Y", "Z"]:
            d.append((f"{p}{q}", [q], p))
    for q in range(n-1):
        for pp in ["ZZ", "XX"]:
            d.append((f"{pp}{q},{q+1}", [q, q+1], pp))
    return d

def stim_spread(nq, depth, ntrial=30, seed=SEED):
    # Stim is a hard requirement here: the operator-spreading boundary is a
    # claim the manuscript makes, so a missing dependency has to stop the run
    # rather than silently produce an empty panel in a figure the text cites.
    import stim
    rng = np.random.default_rng(seed); weights = []
    for _ in range(ntrial):
        c = stim.Circuit()
        for q in range(nq): c.append("H", [q]); c.append("H", [q])  # touch every qubit (identity) so tableau has nq qubits
        for layer in range(depth):
            order = list(range(nq)); rng.shuffle(order)
            for i in range(0, nq-1, 2):
                a, b = order[i], order[i+1]; c.append(rng.choice(["CX","CZ","ISWAP"]), [a, b])
            for q in range(nq):
                if rng.random() < 0.5: c.append(rng.choice(["H","S","SQRT_X"]), [q])
        sim = stim.TableauSimulator(); sim.do(c); t = sim.current_inverse_tableau() ** -1
        p = stim.PauliString(nq); p[0] = 3; ev = t(p)
        weights.append(sum(1 for k in range(nq) if ev[k] != 0))
    return {"depth": depth, "mean_weight": float(np.mean(weights)), "max_weight": int(np.max(weights))}

def local_diagnosis(qubits, word):
    """Run the ACTUAL workflow computation for one dictionary entry on its 1-2 qubit support.

    Builds the per-qubit XY4 schedule on the support, takes the exact toggling average of the entry's
    own generator, and reads it out through the restricted response map. Cost depends only on the
    support size (1 or 2 qubits), never on the total n. Returns (||K||, control_blind, dim ker M)."""
    from pdet_core import Schedule, response_map, kernel_dim
    nq = len(qubits); d = 2 ** nq
    V = _string_op(word, nq)
    K = toggling_average(_xy4_schedule(nq), V)
    scale = float(np.linalg.norm(V))                 # un-decoupled toggling average is V itself (T = 1)
    knorm = float(np.linalg.norm(K))
    blind = knorm <= 1e-9 * scale                    # ideal pulses: the twirl is exact, so this is 0 or O(1)

    # the readout side, on the same support, so the timing covers the whole per-entry workflow
    def stt(v):
        v = np.array(v, complex); v /= np.linalg.norm(v); return np.outer(v, v.conj())
    e = np.eye(d, dtype=complex)
    S = [stt(e[i] + 1j * e[(i + 1) % d]) for i in range(d)] + [stt(e[i]) for i in range(d)]
    O = [_string_op(w, nq) for w in (["Z", "X"] if nq == 1 else ["ZI", "IZ", "ZZ", "XI", "IX"])]
    nstep = 8
    sc = Schedule([np.zeros((d, d), complex) for _ in range(nstep)], dt=1.0 / nstep)
    M = response_map(sc, [K], S, O)
    return knorm, blind, kernel_dim(M)

def main():
    res = {"seed": SEED, "thesis": "locality-scaled K-level diagnosis: O(n) for local dictionary under per-qubit DD"}
    rows = {"n": [], "dict_size": [], "kernel_dim": [], "blind_1local": [], "blind_2local": [], "runtime_ms": []}
    local_diagnosis([0], "Z"); local_diagnosis([0, 1], "ZZ")  # warm up imports/caches before timing
    verdicts = {}
    for n in [2, 3, 4, 5, 6, 7]:
        D = local_dictionary(n)
        reps = []
        for _rep in range(TIMING_REPEATS):
            t0 = time.perf_counter()
            kdim = b1 = b2 = 0
            for name, qubits, ptype in D:
                knorm, blind, _ = local_diagnosis(qubits, ptype)
                verdicts[ptype] = {"K_norm": round(knorm, 12), "control_blind": bool(blind)}
                kdim += 1 if blind else 0
                if len(qubits) == 1: b1 += 1 if blind else 0
                else: b2 += 1 if blind else 0
            reps.append((time.perf_counter() - t0) * 1e3)
        # the reported figure is the median over repeats, so a scheduling hiccup cannot move a published number
        rt = float(np.median(reps))
        rows["n"].append(n); rows["dict_size"].append(len(D)); rows["kernel_dim"].append(kdim)
        rows["blind_1local"].append(b1); rows["blind_2local"].append(b2)
        rows["runtime_ms"].append(round(rt, 2))
    # A wall-clock number is not reproducible to two digits on a shared machine, and its absolute scale says
    # more about the machine than about the algorithm. What IS reproducible, and is the actual claim, is that
    # the cost is linear in n rather than exponential: the fitted slope is recorded for reference, and the
    # coefficient of determination of the linear fit is the quantity the manuscript quotes.
    _n = np.asarray(rows["n"], float); _t = np.asarray(rows["runtime_ms"], float)
    _fit = np.polyfit(_n, _t, 1)
    _res = _t - np.polyval(_fit, _n)
    rows["runtime_slope_ms_per_qubit"] = round(float(_fit[0]), 2)
    rows["runtime_linear_r2"] = round(float(1.0 - _res.dot(_res) / ((_t - _t.mean()) ** 2).sum()), 4)
    rows["runtime_note"] = ("wall clock on one core of a commodity laptop, each point the median of "
                            "%d repeats; the reproducible claim is the linearity, not the absolute scale"
                            % TIMING_REPEATS)
    res["per_direction_verdict"] = verdicts
    res["locality_scaled_kernel"] = rows
    res["scaling_note"] = ("dict size grows O(n) (3n single-qubit + 2(n-1) NN-2-local) and each blind "
                           "classification is a 1-2 qubit toggling computation INDEPENDENT of n, so the K-level "
                           "diagnosis for LOCAL perturbations under per-qubit DD is O(n), not O(2^n). The verdict "
                           "DISCRIMINATES: synchronous per-qubit XY4 twirls every single-qubit Pauli to zero, so "
                           "all 3n single-qubit directions are control-blind, while a nearest-neighbour 2-local "
                           "string is invariant under the simultaneous pulse (Ad_{P x P}(Z x Z) = (+-Z) x (+-Z) = "
                           "Z x Z) and survives the twirl -- the 2(n-1) two-local directions stay visible. That is "
                           "the control lever being direction-specific, computed rather than assumed.")
    # operator-spreading boundary
    res["operator_spreading_boundary"] = {str(d): stim_spread(7, d) for d in [1, 2, 4, 8]}
    res["boundary_note"] = ("Entangling control spreads a local Pauli to high-weight strings: on n=7 the mean "
                            "weight grows with circuit depth -> beyond a depth the local-dictionary assumption "
                            "fails. CLAIM = locality-scaled diagnosis (local errors under local/low-depth control), "
                            "NOT general scalable tomography. This is a stated limitation.")
    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import figstyle; figstyle.apply()
    fig, ax = plt.subplots(1, 2, figsize=figstyle.figsize(1.0, 2.5))
    ax[0].plot(rows["n"], rows["dict_size"], "o-", label="dictionary size O(n)")
    ax[0].plot(rows["n"], rows["kernel_dim"], "s-", label="kernel dim (blind) O(n)")
    ax[0].plot(rows["n"], [2**nn for nn in rows["n"]], "k--", label="2^n (full Hilbert, for ref)")
    ax[0].set_yscale("log"); ax[0].set_xlabel("n qubits"); ax[0].set_ylabel("count (log)")
    ax[0].legend(loc="upper left")
    axr = ax[0].twinx()
    axr.plot(rows["n"], rows["runtime_ms"], "^-", color="tab:green", label="measured runtime (ms)")
    axr.set_ylabel("measured diagnosis runtime (ms)", color="tab:green")
    axr.tick_params(axis="y", labelcolor="tab:green"); axr.legend(loc="lower right")
    sp = res["operator_spreading_boundary"]
    ds = sorted(int(d) for d in sp if "mean_weight" in sp[d])
    if len(ds) < 2:
        raise RuntimeError("operator-spreading panel has %d usable depths; refusing to "
                           "emit a figure the manuscript cites with an empty axis" % len(ds))
    ax[1].plot(ds, [sp[str(d)]["mean_weight"] for d in ds], "o-")
    ax[1].set_xticks(ds)
    ax[1].set_xlabel("entangling circuit depth")
    ax[1].set_ylabel("mean Pauli weight (initially-local, n=7)")
    fig.tight_layout(); figstyle.save(fig, OUT, "fig_scalability")
    with open(os.path.join(OUT, "scalability_results.json"), "w") as f: json.dump(res, f, indent=2, default=str)
    print("\n===== SELF-CHECK L6: scalability (locality-scaled) =====")
    print(" n:", rows["n"]); print(" dict size:", rows["dict_size"]); print(" kernel dim:", rows["kernel_dim"])
    print(" runtime ms:", rows["runtime_ms"], "(O(n), independent of 2^n)")
    print(" operator spreading (n=7) mean weight vs depth:",
          {d: sp[d].get("mean_weight") for d in sp})
    print(" => locality-scaled diagnosis O(n); entangling control = the honest boundary.")
    print("=======================================================\n")
    return res

if __name__ == "__main__":
    main()
