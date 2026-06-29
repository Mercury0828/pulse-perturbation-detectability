"""
PDET Phase-1b -- the GENUINE control-design knob (survives the basis-change audit).

KEY DISTINCTION (defeats the Phase-1a audit's central objection):
  - A perturbation invisible only to a RESTRICTED readout can be exposed by a measurement BASIS change
    (a readout pre-rotation). That is measurement design, NOT a control-schedule contribution (audit: YELLOW).
  - A perturbation whose toggling-frame generator K = integral U0^dag V U0 dt is AVERAGED TO ~0 by the schedule
    is invisible to EVERY measurement (zero first-order signal in all bases). NO basis change can expose it.
    Only changing the DURING-GATE schedule (the K integral) exposes it. THAT is a genuine control-design knob.

Canonical realistic instance: a spin-echo / dynamical-decoupling schedule (used by engineers for NOISE
SUPPRESSION) averages a static Z-detuning to K~0 -> the detuning becomes a DETECTION BLIND SPOT, invisible even to
full tomography. PDET maps this blind-spot subspace and prescribes a minimal during-gate schedule modification
(break the echo symmetry / shift the pi-pulse) that restores K_Z and makes the error finite-shot detectable --
quantifying the PROTECTION vs DETECTABILITY trade-off. The individual facts (DD averages errors; breaking DD
restores them) are KNOWN; PDET's contribution is the systematic per-direction blind-spot map + the minimal-knob
prescription + honest finite-shot accounting.

Run: python phase1b_dd_blindspot.py -> ../results/phase1/{phase1b_results.json, fig_dd_blindspot.png}

"""
from __future__ import annotations
import json, os
import numpy as np
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pdet_core import Schedule, toggling_generator, response_map, singular_spectrum, benign_projector

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "phase1"); os.makedirs(OUT, exist_ok=True)
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]], complex); Z = np.array([[1, 0], [0, -1]], complex)
N, T = 60, 60.0; DT = T / N; SEED = 20260628

def st(v): v = np.array(v, complex); v = v / np.linalg.norm(v); return np.outer(v, v.conj())

# Exact two-segment spin echo: free (H0=0) for t1=f*T, IDEAL instantaneous pi-X, free for t2=(1-f)*T.
# Toggling generator of operator Vop:  K = t1*Vop + t2*(X^dag Vop X).  U0(T) = X if a pi was applied else I.
# Static Z-detuning:  K_Z = (t1 - t2) Z = (2f-1) T Z  -> EXACTLY 0 at the symmetric echo f=0.5 (a true null).
def two_segment_K(Vop, f, has_pi):
    if not has_pi:
        return T * Vop, np.eye(2, dtype=complex)
    t1, t2 = f * T, (1 - f) * T
    K = t1 * Vop + t2 * (X.conj().T @ Vop @ X)
    return K, X  # U0(T)=X

def full_tomo_access():
    S = [st([1, 0]), st([0, 1]), st([1, 1]), st([1, -1]), st([1, 1j]), st([1, -1j])]
    O = [X, Y, Z]
    return S, O

def response_two_segment(f, has_pi, S, O):
    K, U0T = two_segment_K(Z, f, has_pi)
    Otil = [U0T.conj().T @ Oo @ U0T for Oo in O]
    rows = []
    for Ot in Otil:
        for rho in S:
            rows.append(-1j * np.trace((rho @ Ot - Ot @ rho) @ K))
    M = np.array(rows, dtype=complex).reshape(-1, 1)
    assert np.max(np.abs(M.imag)) < 1e-9 * (np.max(np.abs(M.real)) + 1e-15)
    return M.real, float(np.linalg.norm(K))

def margin_and_cost(f, has_pi, S, O, theta_mag=0.05, V=1.0, fa=0.05, miss=0.05):
    M, kZ = response_two_segment(f, has_pi, S, O)
    margin = float(np.linalg.norm(M[:, 0]) * theta_mag)
    if margin <= 1e-12:
        Nstar = np.inf
    else:
        z = norm.ppf(1 - fa) + norm.ppf(1 - miss); Nstar = V * z ** 2 / margin ** 2
    return kZ, margin, Nstar

def montecarlo(f, has_pi, S, O, theta_mag, Nshots, V=1.0, n_rep=400, seed=SEED):
    rng = np.random.default_rng(seed)
    M, _ = response_two_segment(f, has_pi, S, O)
    v_att = M[:, 0] * theta_mag; v_ben = np.zeros_like(v_att)
    diff = v_att - v_ben; nrm = np.linalg.norm(diff) + 1e-18; u = diff / nrm
    thr = 0.5 * (u @ v_att + u @ v_ben); sigma = np.sqrt(V / Nshots)
    fa = miss = 0
    for _ in range(n_rep):
        if u @ (v_ben + sigma * rng.standard_normal(v_ben.shape)) > thr: fa += 1
        if u @ (v_att + sigma * rng.standard_normal(v_att.shape)) <= thr: miss += 1
    return fa / n_rep, miss / n_rep

def main():
    S, O = full_tomo_access()
    res = {"seed": SEED, "access": "FULL single-qubit tomography (6 states x XYZ)", "target": "static Z-detuning"}

    # sweep the pi-pulse position: 0.5 = symmetric echo (blind spot); away from 0.5 = diagnostic knob; free
    fracs = [("free", None), (0.5, True), (0.45, True), (0.40, True), (0.33, True), (0.25, True), (0.10, True)]
    sweep = []
    for tag, has in fracs:
        f = 0.5 if tag == "free" else tag
        kZ, margin, Nstar = margin_and_cost(f, has is True, S, O)
        sweep.append({"pi_frac": tag, "K_Z_norm": kZ, "margin": margin,
                      "shot_cost_Nstar": (None if not np.isfinite(Nstar) else Nstar),
                      "Nstar_str": ("INF" if not np.isfinite(Nstar) else round(Nstar, 1))})
    res["sweep"] = sweep

    # decisive comparison: symmetric echo (blind) vs broken echo (knob) -- with Monte-Carlo at predicted N*
    kZe, me, Ne = margin_and_cost(0.5, True, S, O)
    kZk, mk, Nk = margin_and_cost(0.33, True, S, O)
    Ntest = int(min(Nk * 2, 1e6)) if np.isfinite(Nk) else int(1e6)
    fae, mie = montecarlo(0.5, True, S, O, 0.05, int(1e6))            # echo: try hard (1e6 shots) -> still fails
    fak, mik = montecarlo(0.33, True, S, O, 0.05, Ntest)
    res["decisive"] = {
        "echo_symmetric": {"K_Z_norm": kZe, "margin": me, "Nstar": ("INF" if not np.isfinite(Ne) else Ne),
                           "montecarlo_at_1e6_shots": {"FA": fae, "MISS": mie}},
        "knob_broken_echo_frac0.33": {"K_Z_norm": kZk, "margin": mk, "Nstar": round(Nk, 1),
                                      "montecarlo_at_Ntest": {"N": Ntest, "FA": fak, "MISS": mik}},
        "differentiator": ("Echo blind spot is invisible to FULL tomography (Nstar=INF; 1e6 shots still random). "
                           "NO measurement-basis change can fix it (K_Z~0 => zero signal in ALL bases). Only the "
                           "DURING-GATE schedule change (break echo symmetry) restores K_Z and finite-shot "
                           "detectability. This is a genuine control-design knob, not a readout-basis change."),
        "knob_is_control_not_measurement": bool((not np.isfinite(Ne)) and np.isfinite(Nk))}

    _fig(res);
    with open(os.path.join(OUT, "phase1b_results.json"), "w") as f: json.dump(res, f, indent=2, default=str)
    print("\n===== Phase-1b: genuine control-design knob (DD blind spot) =====")
    for s in sweep: print(f"  pi_frac={str(s['pi_frac']):5s}: ||K_Z||={s['K_Z_norm']:.3f} margin={s['margin']:.3f} N*={s['Nstar_str']}")
    print(" DECISIVE:")
    print(f"  symmetric echo: ||K_Z||={kZe:.3f} margin={me:.3e} N*={'INF' if not np.isfinite(Ne) else round(Ne,1)}; MC@1e6: FA={fae:.3f} MISS={mie:.3f}")
    print(f"  broken echo(0.33): ||K_Z||={kZk:.3f} margin={mk:.3f} N*={round(Nk,1)}; MC@{Ntest}: FA={fak:.3f} MISS={mik:.3f}")
    print(f"  knob_is_control_not_measurement = {res['decisive']['knob_is_control_not_measurement']}")
    print("================================================================\n")
    return res

def _fig(res):
    sw = res["sweep"]; xs = [str(s["pi_frac"]) for s in sw]
    kZ = [s["K_Z_norm"] for s in sw]; mg = [s["margin"] for s in sw]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].bar(xs, kZ, color="C0"); ax[0].set_title("||K_Z|| vs pi-pulse position (0.5=symmetric echo=blind spot)")
    ax[0].set_xlabel("pi-pulse fractional position"); ax[0].set_ylabel("||K_Z|| (toggling-frame integral)")
    nstars = [s["shot_cost_Nstar"] for s in sw]
    bars = [1e8 if n is None else n for n in nstars]
    ax[1].bar(xs, bars, color=["gray" if n is None else "C2" for n in nstars]); ax[1].set_yscale("log")
    ax[1].set_title("finite-shot cost N* for Z-detuning (gray=INF: invisible to ALL measurements)")
    ax[1].set_xlabel("pi-pulse fractional position"); ax[1].set_ylabel("N* (log)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_dd_blindspot.png"), dpi=120); plt.close(fig)

if __name__ == "__main__":
    main()
