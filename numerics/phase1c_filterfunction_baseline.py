"""
PDET Phase-1c -- R-O5: filter-function / DD-noise-spectroscopy baseline, and EXACTLY what PDET adds beyond it.

Honest connection: for a STATIC (DC) coherent perturbation V, the first-order filter-function amplitude at omega=0
is  F_V(0) = ||K_V||^2,  K_V = integral U0^dag V U0 dt  -- the SAME toggling-frame quantity PDET uses. So the
filter-function/DD-spectroscopy baseline already captures K-LEVEL (in)visibility, including the DD blind spot
(F=0 when the schedule averages V away). PDET must NOT claim that as new.

What PDET adds (the diagnostic design layer):
  (1) READOUT-LEVEL invisibility: a direction with K != 0 (filter function says "sensitive") can STILL be
      first-order invisible under the engineer's ACTUAL restricted (S,O) because [rho_s, Otilde_l] is orthogonal
      to K. Filter-function sensitivity assumes an optimal/coherence readout; PDET's kernel ker M makes the
      (S,O)-dependence explicit and catches this case. Filter functions MISS it.
  (2) FINITE-SHOT DETECTION COST: PDET returns a shot budget N* (FA/MISS), not a spectral sensitivity.
  (3) UNIFIED kernel over the whole perturbation dictionary + minimal schedule/readout augmentation prescription.

Two sources of invisibility, one framework:
  K-level (filter-function / DD blind spot)  -- fix by CONTROL (un-average the schedule).
  readout-level (restricted (S,O))           -- fix by MEASUREMENT/probe design.
PDET's ker M = (filter-function K-visibility) intersect (readout access) + finite-shot.

Also: a genuine MULTI-QUBIT DD blind spot -- a control-echo cancels K_ZI AND K_ZZ (a realistic ZZ spectator
coherent error), hidden from ALL measurements; breaking the echo exposes it.

Run: python phase1c_filterfunction_baseline.py -> ../results/phase1/phase1c_results.json + table printout.

"""
from __future__ import annotations
import json, os
import numpy as np
from scipy.stats import norm
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "phase1"); os.makedirs(OUT, exist_ok=True)
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]], complex); Z = np.array([[1, 0], [0, -1]], complex)
def kron(*a):
    r = np.array([[1]], complex)
    for x in a: r = np.kron(r, x)
    return r
def st(v): v = np.array(v, complex); v = v / np.linalg.norm(v); return np.outer(v, v.conj())
def dag(A): return A.conj().T

# two-segment toggling: free t1, ideal pi (rotation R) at fraction f, free t2.  K = t1 V + t2 R^dag V R.
def K_two_segment(V, f, R):
    if R is None: return 1.0 * V                      # free, T=1
    t1, t2 = f, 1 - f
    return t1 * V + t2 * (dag(R) @ V @ R), (R if True else None)

def K_and_U0(V, f, R):
    if R is None: return 1.0 * V, np.eye(V.shape[0], dtype=complex)
    return f * V + (1 - f) * (dag(R) @ V @ R), R

def response(V, f, R, S, O):
    K, U0 = K_and_U0(V, f, R)
    Otil = [dag(U0) @ Oo @ U0 for Oo in O]
    rows = [(-1j * np.trace((rho @ Ot - Ot @ rho) @ K)) for Ot in Otil for rho in S]
    M = np.array(rows, dtype=complex).reshape(-1, 1)
    return M.real, float(np.linalg.norm(K))   # margin-vector, ||K||

def filter_function_dc(V, f, R):
    K, _ = K_and_U0(V, f, R)
    return float(np.linalg.norm(K) ** 2)      # F_V(0) = ||K_V||^2

def Nstar(margin, V=1.0, fa=0.05, miss=0.05):
    if margin <= 1e-12: return np.inf
    z = norm.ppf(1 - fa) + norm.ppf(1 - miss); return V * z ** 2 / margin ** 2

def main():
    res = {}

    # ---------- 1q: filter-function vs PDET, two readout budgets ----------
    dirs = {"X(amp)": X, "Y(phase)": Y, "Z(detuning)": Z}
    S = [st([1, 0]), st([0, 1]), st([1, 1]), st([1, 1j])]
    O_full = [X, Y, Z]; O_Z = [Z]
    table = []
    for nm, V in dirs.items():
        ff = filter_function_dc(V, None, None)              # free schedule, DC filter function
        m_full, _ = response(V, None, None, S, O_full); m_full = np.linalg.norm(m_full) * 0.05
        m_Z, _ = response(V, None, None, S, O_Z); m_Z = np.linalg.norm(m_Z) * 0.05
        table.append({"direction": nm, "filter_F0(=||K||^2)": round(ff, 3),
                      "FF_says_sensitive": bool(ff > 1e-9),
                      "PDET_margin_full_readout": round(m_full, 3), "PDET_visible_full": bool(m_full > 1e-9),
                      "PDET_margin_Zonly_readout": round(m_Z, 4), "PDET_visible_Zonly": bool(m_Z > 1e-9),
                      "Nstar_Zonly": ("INF" if not np.isfinite(Nstar(m_Z)) else round(Nstar(m_Z), 1)),
                      "PDET_catches_readout_invisibility_FF_misses":
                          bool(ff > 1e-9 and m_Z <= 1e-9)})
    res["one_qubit_FF_vs_PDET"] = table

    # ---------- DD blind spot (K-level): filter function ALSO catches it (both agree) ----------
    ff_echo = filter_function_dc(Z, 0.5, X); ff_broken = filter_function_dc(Z, 0.33, X)
    res["dd_blindspot_1q"] = {"F0_symmetric_echo": round(ff_echo, 4), "F0_broken_echo": round(ff_broken, 3),
                              "note": "DD blind spot is K-level: filter function ALSO sees F0=0 at symmetric echo. "
                                      "PDET does not claim this as new; it adds readout-level invisibility + finite-shot."}

    # ---------- 2q multi-qubit DD blind spot: control-echo hides ZI AND ZZ spectator ----------
    Rc = kron(X, I)
    two = {}
    for nm, V in {"ZI(ctrl detuning)": kron(Z, I), "ZZ(spectator)": kron(Z, Z),
                  "IX(crosstalk)": kron(I, X), "IZ(tgt detuning)": kron(I, Z)}.items():
        two[nm] = {"F0_echo": round(filter_function_dc(V, 0.5, Rc), 4),
                   "F0_broken": round(filter_function_dc(V, 0.33, Rc), 3),
                   "F0_free": round(filter_function_dc(V, None, None), 3)}
    res["dd_blindspot_2q_control_echo"] = {
        "per_direction": two,
        "note": "Control-echo (pi-X on q0) averages K_ZI AND K_ZZ to 0 (a realistic ZZ spectator coherent error "
                "hidden from ALL measurements). Breaking the echo (during-gate control change) exposes them -- a "
                "genuine MULTI-QUBIT control-design knob. IX/IZ on the target are unaffected (not echoed)."}

    # ---------- what PDET adds, stated ----------
    res["pdet_delta_vs_filter_function"] = {
        "K_level_invisibility": "captured by BOTH filter functions and PDET (DD blind spot). NOT a PDET novelty.",
        "readout_level_invisibility": "K!=0 but invisible under restricted (S,O); PDET catches, filter functions MISS.",
        "finite_shot_cost": "PDET returns detection shot budget N* (FA/MISS); filter functions give spectral sensitivity.",
        "unified_workflow": "PDET = kernel over the dictionary under real (S,O) + minimal control/readout augmentation."}

    with open(os.path.join(OUT, "phase1c_results.json"), "w") as f: json.dump(res, f, indent=2, default=str)
    _print(res); return res

def _print(r):
    print("\n===== Phase-1c: filter-function baseline vs PDET (R-O5) =====")
    print(" 1q: filter-function DC sensitivity vs PDET restricted-readout detectability")
    for t in r["one_qubit_FF_vs_PDET"]:
        print(f"  {t['direction']:12s}: F0={t['filter_F0(=||K||^2)']:.3f} (FF sensitive={t['FF_says_sensitive']}) | "
              f"PDET full={t['PDET_margin_full_readout']:.3f}(vis={t['PDET_visible_full']}) "
              f"Z-only={t['PDET_margin_Zonly_readout']:.4f}(vis={t['PDET_visible_Zonly']}, N*={t['Nstar_Zonly']}) | "
              f"PDET-catches-readout-invis-FF-misses={t['PDET_catches_readout_invisibility_FF_misses']}")
    print(f" DD blind spot 1q: F0(echo)={r['dd_blindspot_1q']['F0_symmetric_echo']} "
          f"F0(broken)={r['dd_blindspot_1q']['F0_broken_echo']}  (both FF & PDET see it -> not a PDET novelty)")
    print(" 2q control-echo DD blind spot:")
    for nm, v in r["dd_blindspot_2q_control_echo"]["per_direction"].items():
        print(f"   {nm:20s}: F0 echo={v['F0_echo']} broken={v['F0_broken']} free={v['F0_free']}")
    print("=============================================================\n")

if __name__ == "__main__":
    main()
