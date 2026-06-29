"""
PDET Phase-1d -- R-O2: active / Fisher experiment-design baseline (Granade 1207.1655, Hincks 1806.02427),
the closest conceptual competitor to the control knob.

Fisher/active experiment design optimizes the MEASUREMENT / PROBE (and which experiment to run next) to maximize
information about a target parameter. Key boundary, made concrete here:
  - QFI (max over ALL probes/measurements) for estimating theta along direction V equals 4 * max_psi Var_psi(L_V)
    = (spectral diameter of L_V)^2, with L_V = K_V (toggling-frame generator).  For a K-LEVEL blind spot
    (K_V = 0, e.g. a symmetric-echo DD null) the QFI is EXACTLY 0: NO measurement and NO probe can detect it.
    Fisher/active experiment design therefore CANNOT expose a DD blind spot -- only changing the during-gate
    schedule (the control knob) restores QFI > 0.
  - For READOUT-LEVEL invisibility (K_V != 0 but invisible under the current restricted O), QFI > 0: an optimal
    measurement EXISTS, so Fisher design CAN fix it (equivalent to a basis change). PDET's kernel says WHICH
    directions are which, and which lever (control vs measurement) each needs.

So: PDET's control knob does something Fisher/active design provably cannot (expose K=0 directions); for
readout-level cases the two agree that a measurement change suffices. This is the honest placement of the baseline.

Run: python phase1d_fisher_baseline.py -> ../results/phase1/phase1d_results.json + printout.
"""
from __future__ import annotations
import json, os
import numpy as np
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "phase1"); os.makedirs(OUT, exist_ok=True)
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]], complex); Z = np.array([[1, 0], [0, -1]], complex)
def dag(A): return A.conj().T

def K_two_seg(V, f, R):
    if R is None: return 1.0 * V
    return f * V + (1 - f) * (dag(R) @ V @ R)

def qfi_max_over_probes(K):
    """QFI (max over probe states & measurements) for estimating theta along a direction with generator K.
       = (lambda_max - lambda_min)^2 of the Hermitian generator K (the optimal-probe QFI, Pang-Brun)."""
    ev = np.linalg.eigvalsh((K + dag(K)) / 2)   # K is Hermitian for Hermitian V; symmetrize for safety
    return float((ev[-1] - ev[0]) ** 2)

def cfi_restricted(K, S, O):
    """Classical Fisher info available under restricted (S,O): proxy = sum over (s,l) of (first-order signal)^2
       (the achievable restricted-measurement information; 0 iff invisible under (S,O))."""
    tot = 0.0
    for rho in S:
        for Ob in O:
            sig = (-1j * np.trace((rho @ Ob - Ob @ rho) @ K)).real
            tot += sig ** 2
    return float(tot)

def st(v): v = np.array(v, complex); v = v / np.linalg.norm(v); return np.outer(v, v.conj())

def main():
    S = [st([1, 0]), st([0, 1]), st([1, 1]), st([1, 1j])]
    O_full = [X, Y, Z]; O_Z = [Z]
    res = {}

    # free schedule: QFI (max over measurement) vs CFI under restricted readouts
    rows = []
    for nm, V in {"X(amp)": X, "Y(phase)": Y, "Z(detuning)": Z}.items():
        K = K_two_seg(V, None, None)
        qfi = qfi_max_over_probes(K)
        rows.append({"direction": nm, "QFI_max_over_measurements": round(qfi, 3),
                     "CFI_full_readout": round(cfi_restricted(K, S, O_full), 3),
                     "CFI_Zonly_readout": round(cfi_restricted(K, S, O_Z), 4),
                     "fisher_design_can_fix": bool(qfi > 1e-9),
                     "needs": ("control-knob (K=0: QFI=0, no measurement helps)" if qfi <= 1e-9
                               else "measurement/probe design (QFI>0: optimal measurement exists)")})
    res["free_schedule"] = rows

    # DD blind spot: symmetric echo makes QFI=0 for Z-detuning -> Fisher design provably fails; control knob fixes.
    K_echo = K_two_seg(Z, 0.5, X); K_broken = K_two_seg(Z, 0.33, X)
    res["dd_blindspot"] = {
        "QFI_Z_symmetric_echo": round(qfi_max_over_probes(K_echo), 6),
        "QFI_Z_broken_echo": round(qfi_max_over_probes(K_broken), 3),
        "verdict": ("At the symmetric echo QFI(Z-detuning)=0 over ALL measurements/probes => Fisher/active "
                    "experiment design CANNOT detect it. Breaking the echo (control knob) restores QFI>0. This is "
                    "a capability the measurement-optimization baseline provably lacks.")}

    res["placement"] = {
        "fisher_active_design_handles": "readout-level invisibility (QFI>0): finds the optimal measurement/probe.",
        "fisher_active_design_cannot_handle": "K-level invisibility / DD blind spot (QFI=0): no measurement helps.",
        "pdet_adds": "the kernel that classifies each direction (control- vs measurement-fixable) + the control "
                     "knob for the K-level case + finite-shot detection cost."}

    with open(os.path.join(OUT, "phase1d_results.json"), "w") as f: json.dump(res, f, indent=2, default=str)
    print("\n===== Phase-1d: Fisher / active-design baseline (R-O2) =====")
    for r in rows:
        print(f"  {r['direction']:12s}: QFI(max meas)={r['QFI_max_over_measurements']:.3f} "
              f"CFI_full={r['CFI_full_readout']:.3f} CFI_Zonly={r['CFI_Zonly_readout']:.4f} -> needs {r['needs']}")
    print(f"  DD blind spot: QFI(Z, symmetric echo)={res['dd_blindspot']['QFI_Z_symmetric_echo']} "
          f"(=0 => Fisher design FAILS); QFI(broken)={res['dd_blindspot']['QFI_Z_broken_echo']} (control knob restores)")
    print("===========================================================\n")
    return res

if __name__ == "__main__":
    main()
