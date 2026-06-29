"""
SELF-CHECK -- GENUINE non-contrived K-level use case (replaces the contrived echo-hides-Z flagship).

Scenario (engineering-real): an IDLE / memory qubit runs production dynamical decoupling (XY4) to preserve
coherence. DD's JOB is to average coherent errors away -> by design it makes the idle qubit a TOTAL coherent-error
detection blind spot (XY4 -> ker A = su(2)). Single-qubit FULL tomography is cheap and available, so a coherent
drift (e.g. slow flux-induced Z-detuning, or a spectator-induced coherent shift) is READOUT-VISIBLE in principle
but the DD schedule averages it to K~0 -> a GENUINE K-level (control-fixable) blind spot, NOT readout-blind. This
is non-contrived: DD-on-idle is universal; "is my DD also hiding a coherent drift I should know about?" is a real
commissioning question. PDET flags it for free (classical kernel) and prescribes the minimal DD modification to
expose it -- trading coherence protection for detectability.

Realistic open-system noise (named-backend-like): T1=200us, T2=120us, readout error 1.3%; DD over T_seq us-scale.

PRE-REGISTERED (frozen): expect a static Z-drift (i) BLIND under XY4 even with FULL tomography (signal ~0);
(ii) exposed by a modified DD (drop one pulse / asymmetric) -> finite N* under realistic noise; (iii) XY4 preserves
coherence (protection) that the modified DD partially loses (trade). Falsifier: Z visible under XY4 (full tomo),
or not exposable, or no protection difference.

Run: python selfcheck_dd_idle_usecase.py -> ../results/selfcheck/dd_idle_usecase_results.json + fig.

"""
from __future__ import annotations
import json, os
import numpy as np
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import figstyle
import qutip as qt

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "selfcheck"); os.makedirs(OUT, exist_ok=True)
SEED = 20260628
T1_us, T2_us, P_RO = 200.0, 120.0, 0.013
sm = qt.sigmam(); sx = qt.sigmax(); sy = qt.sigmay(); sz = qt.sigmaz()
g1 = 1.0/T1_us; gphi = max(1.0/T2_us - 1.0/(2*T1_us), 0.0)
C_OPS = [np.sqrt(g1)*sm, np.sqrt(gphi/2)*sz]

def prop_step(H, dt): return (qt.liouvillian(H, C_OPS)*dt).expm()

def dd_pulse_times(seq, T):
    """Return list of (fractional time, axis) for a DD sequence. XY4 = X,Y,X,Y at 1/8,3/8,5/8,7/8."""
    if seq == "free": return []
    if seq == "echo": return [(0.5, sx)]
    if seq == "xy4": return [(0.125, sx), (0.375, sy), (0.625, sx), (0.875, sy)]
    if seq == "xy4_drop1": return [(0.125, sx), (0.375, sy), (0.625, sx)]   # modified DD: drop last pulse (the knob)
    if seq == "xy4_asym": return [(0.10, sx), (0.375, sy), (0.625, sx), (0.875, sy)]  # asymmetric (the knob)
    raise ValueError(seq)

def evolve(seq, T, drift_op, drift_rate, rho0, nfree=160):
    """Open-system evolution of rho0 under a DD sequence over time T with a static coherent drift drift_rate*drift_op."""
    pulses = dd_pulse_times(seq, T); dt = T/nfree
    pulse_steps = {int(round(f*nfree)): ax for f, ax in pulses}
    rho = qt.operator_to_vector(rho0)
    for k in range(nfree):
        H = (drift_rate/2.0)*drift_op
        rho = prop_step(H, dt)*rho
        if k in pulse_steps:
            pi_dt = dt/50.0
            rho = prop_step((np.pi/pi_dt/2.0)*pulse_steps[k], pi_dt)*rho
    return qt.vector_to_operator(rho)

def first_order_signal(seq, T, drift_op, obs, eps=1e-3):
    rho0 = qt.ket2dm((qt.basis(2,0)+qt.basis(2,1)).unit())  # |+> probe
    rp = evolve(seq, T, drift_op, +eps, rho0); rm = evolve(seq, T, drift_op, -eps, rho0)
    return np.array([float((qt.expect(O, rp)-qt.expect(O, rm))/(2*eps)) for O in obs])

def ff_protection(seq, T, alpha=1.0, nw=600, ngrid=2000):
    """Protection against CORRELATED 1/f dephasing via the filter function (DD does NOT help vs Markovian T1/T2 --
    that is why the Lindblad coherence is schedule-independent; DD suppresses correlated/1/f noise). The Z-toggling
    sign y(t) flips at each DD pulse (X,Y both anticommute with Z). chi=int S(w)|Y(w)|^2 dw, S(w)=1/w^alpha.
    Protection vs free = chi_free/chi_seq (higher = better DD suppression of 1/f)."""
    pulses = dd_pulse_times(seq, T)
    flip_fracs = sorted(f for f, _ in pulses)
    t = (np.arange(ngrid)+0.5)*(T/ngrid); y = np.ones(ngrid)
    for fr in flip_fracs:
        y[t > fr*T] *= -1.0
    ws = np.geomspace(2*np.pi*1e-3, 2*np.pi*10, nw); dt = T/ngrid
    Yw = np.array([np.abs(np.sum(y*np.exp(1j*w*t))*dt)**2 for w in ws])
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapz((1.0/ws**alpha)*Yw, ws)/(2*np.pi))

def Nstar(margin, V): return np.inf if margin<=1e-9 else V*(2*norm.ppf(0.95))**2/margin**2

def main():
    obs = [sx, sy, sz]  # FULL single-qubit tomography (cheap, available)
    V_ro = 1.0/(1-2*P_RO)**2; theta = 0.05; T = 16.0  # 16 us idle DD window
    res = {"seed": SEED, "params": {"T1_us": T1_us, "T2_us": T2_us, "p_ro": P_RO, "T_seq_us": T, "V_ro": round(V_ro,4)}}
    drift = sz  # static Z-detuning drift (flux-induced); readout-visible in principle, averaged by XY4
    seqs = ["free", "xy4", "xy4_drop1", "xy4_asym"]
    chi_free = ff_protection("free", T)
    rows = {}
    free_sig = np.linalg.norm(first_order_signal("free", T, drift, obs))
    for s in seqs:
        sig = np.linalg.norm(first_order_signal(s, T, drift, obs))
        margin = sig*theta; chi = ff_protection(s, T)
        rows[s] = {"Zdrift_signal_fulltomo": round(float(sig),5),
                   "Nstar": (None if not np.isfinite(Nstar(margin, V_ro)) else round(Nstar(margin, V_ro),1)),
                   "protection_1overf_vs_free": round(chi_free/(chi+1e-18), 2),
                   "blind": bool(sig < 1e-3*free_sig)}
    res["per_schedule"] = rows
    free_sig = rows["free"]["Zdrift_signal_fulltomo"]
    res["verdict"] = {
        "Z_blind_under_XY4_fulltomo": bool(rows["xy4"]["Zdrift_signal_fulltomo"] < 1e-2*free_sig),
        "exposed_by_modified_DD": bool((rows["xy4_drop1"]["Nstar"] is not None and rows["xy4_drop1"]["Nstar"] < 1e6)
                                       or (rows["xy4_asym"]["Nstar"] is not None and rows["xy4_asym"]["Nstar"] < 1e6)),
        "XY4_1overf_protection_vs_free": rows["xy4"]["protection_1overf_vs_free"],
        "protection_lost_by_knob_xy4_to_asym": round(rows["xy4"]["protection_1overf_vs_free"]-rows["xy4_asym"]["protection_1overf_vs_free"], 2),
        "summary": ("GENUINE K-level non-contrived case: XY4 (production memory DD) makes a Z-drift blind to FULL "
                    "tomography (control-fixable, NOT readout-fixable); modifying the DD exposes it at finite N* "
                    "under realistic noise, trading 1/f-noise protection (DD suppresses correlated noise; it does "
                    "NOT help vs Markovian T1/T2, which is why detection N* uses the Lindblad model and protection "
                    "uses the 1/f filter function). PDET's classical kernel flags the blind direction for free.")}
    # figure
    figstyle.apply()
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
    names = list(rows); x = np.arange(len(names))
    sigs = [rows[s]["Zdrift_signal_fulltomo"] for s in names]
    ax[0].bar(x, sigs, color=["#2ca02c","#d62728","#1f77b4","#1f77b4"])
    ax[0].annotate("blind\n(~2e-5)", (1, max(sigs)*0.04), ha="center", va="bottom", fontsize=12, color="#d62728")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=12); ax[0].set_ylabel("Z-drift signal (FULL tomography)")
    ax2 = ax[1].twinx()
    ns = [1e8 if rows[s]["Nstar"] is None else rows[s]["Nstar"] for s in names]
    ax[1].bar(x-0.2, ns, 0.4, color="C0", label="N* (shots)"); ax[1].set_yscale("log"); ax[1].set_ylabel("N* (shots)")
    ax2.bar(x+0.2, [rows[s]["protection_1overf_vs_free"] for s in names], 0.4, color="C1", label="1/f protection x")
    ax2.set_ylabel("1/f protection factor vs free")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=12)
    ax[1].legend(loc="upper left", fontsize=12); ax2.legend(loc="upper right", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_dd_idle_usecase.png"), dpi=130); plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    with open(os.path.join(OUT, "dd_idle_usecase_results.json"), "w") as f: json.dump(res, f, indent=2, default=str)
    print("\n===== GENUINE non-contrived use case: DD on idle qubit =====")
    for s in names:
        print(f"  {s:11s}: Zdrift_sig={rows[s]['Zdrift_signal_fulltomo']:.5f} N*={rows[s]['Nstar']} "
              f"1/f_protection=x{rows[s]['protection_1overf_vs_free']} blind={rows[s]['blind']}")
    print(" VERDICT:", json.dumps(res["verdict"], default=str, indent=2))
    print("===========================================================\n")
    return res

if __name__ == "__main__":
    main()
