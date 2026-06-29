"""
SELF-CHECK L1 (device realism) -- the biggest TQE gap: does the PDET detectability story SURVIVE realistic
open-system noise (T1/T2 + readout/SPAM), not just closed-system coherent models?

NAMED-BACKEND-LIKE params (IBM Heron / ibm_torino-class, public calibration ranges, 2024-2025):
  T1 = 200 us, T2 = 120 us, single-qubit readout error p_ro = 1.3%, (DD sequence timescale T_seq swept us-scale).
Lindblad collapse ops: sqrt(1/T1) sigma_-, sqrt(gamma_phi/2) sigma_z with gamma_phi = 1/T2 - 1/(2 T1).

PRE-REGISTERED expectation + falsifier (frozen BEFORE running):
  (i)  K-level blind spot PERSISTS: under a symmetric echo, a static Z-detuning has K_Z=0 -> ZERO first-order
       coherent signal; decoherence is theta-INDEPENDENT, so it cannot create a first-order signal where K=0.
       Expect: first-order detuning signal under echo stays ~0 (<= noise floor) WITH T1/T2.  Falsifier: a nonzero
       first-order signal appears at the echo (would mean decoherence creates coherent detectability -- it must not).
  (ii) VISIBLE directions: margin damped by ~exp(-T_seq/T2); finite-shot variance inflated by readout error
       (V_eff = V/(1-2 p_ro)^2). Expect N* up by a realistic factor, still finite.  Falsifier: margin collapses
       to ~0 for a genuinely visible direction at us-scale (would kill detectability under noise).
  (iii) Knob still works: breaking the echo restores K_Z!=0 -> finite-shot-detectable against the decoherence
       background. Falsifier: knob-exposed direction undetectable at a realistic budget (1e6 shots).

Run: python selfcheck_realistic_noise.py -> ../results/selfcheck/realistic_noise_results.json + fig.

"""
from __future__ import annotations
import json, os
import numpy as np
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import figstyle; figstyle.apply()
import qutip as qt

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "selfcheck"); os.makedirs(OUT, exist_ok=True)
SEED = 20260628

# named-backend-like params (us, MHz)
T1_us, T2_us, P_RO = 200.0, 120.0, 0.013
sm = qt.sigmam(); sz = qt.sigmaz(); sx = qt.sigmax(); sy = qt.sigmay(); I2 = qt.qeye(2)
gamma1 = 1.0 / T1_us
gamma_phi = max(1.0 / T2_us - 1.0 / (2 * T1_us), 0.0)
C_OPS = [np.sqrt(gamma1) * sm, np.sqrt(gamma_phi / 2.0) * sz]

def step_liouvillian_prop(H, dt):
    """expm(L dt) as a superoperator (qutip) for a piecewise-constant H with the fixed collapse ops."""
    L = qt.liouvillian(H, C_OPS)
    return (L * dt).expm()

def propagate(step_hams, dt, rho0):
    """Open-system propagation of rho0 through the piecewise-constant schedule (us units; H in rad/us)."""
    rho = qt.operator_to_vector(rho0)
    for H in step_hams:
        rho = step_liouvillian_prop(H, dt) * rho
    return qt.vector_to_operator(rho)

def echo_schedule(T_seq_us, f, has_pi, nfree=40):
    """Free evolution (H0=0) for f*T, ideal pi-X, free for (1-f)*T. Detuning perturbation added separately.
       Returns a function builder(theta) -> (step_hams, dt) with a static Z-detuning of angular rate theta (rad/us)."""
    n1 = max(1, int(round(f * nfree))); n2 = max(1, nfree - n1)
    dt = T_seq_us / (n1 + n2)
    pi_dt = dt / 50.0  # near-instantaneous pi
    def builder(theta):
        H = []
        for _ in range(n1):
            H.append((theta / 2.0) * sz)          # detuning during free evolution
        if has_pi:
            H.append((np.pi / pi_dt / 2.0) * sx)  # one strong short pi-X step (handled with its own dt below)
        for _ in range(n2):
            H.append((theta / 2.0) * sz)
        return H, dt, (pi_dt if has_pi else None), n1
    return builder

def propagate_with_pi(builder, theta):
    H, dt, pi_dt, n1 = builder(theta)
    rho = qt.operator_to_vector(qt.ket2dm((qt.basis(2,0)+qt.basis(2,1)).unit()))  # |+> probe (sensitive to Z)
    idx = 0
    for k, Hk in enumerate(H):
        this_dt = dt
        if pi_dt is not None and k == n1:   # the pi step uses its short dt
            this_dt = pi_dt
        rho = step_liouvillian_prop(Hk, this_dt) * rho
    return qt.vector_to_operator(rho)

def first_order_signal(builder, obs_list, eps=1e-3):
    """Finite-difference first-order signal vector of the Z-detuning (rate eps rad/us) under OPEN-system noise."""
    rp = propagate_with_pi(builder, +eps); rm = propagate_with_pi(builder, -eps)
    return np.array([float(((qt.expect(O, rp) - qt.expect(O, rm)) / (2 * eps))) for O in obs_list])

def readout_inflated_V(base_V=1.0, p_ro=P_RO):
    """Readout error inflates the per-shot variance of a bounded observable: V_eff = base_V / (1-2 p_ro)^2."""
    return base_V / (1 - 2 * p_ro) ** 2

def Nstar(margin, V):
    if margin <= 1e-9: return np.inf
    return V * (2 * norm.ppf(0.95)) ** 2 / margin ** 2

def main():
    res = {"seed": SEED, "params": {"T1_us": T1_us, "T2_us": T2_us, "p_readout": P_RO},
           "preregistered": "see module docstring (expectation + falsifier frozen before run)"}
    obs = [sx, sy, sz]  # full single-qubit tomography
    theta_phys_rate = 0.05  # detuning amplitude used for margin (rad/us); margin uses eps internally
    V_eff = readout_inflated_V()

    # sweep DD sequence time; compare echo (blind) vs broken echo (knob) vs free, closed vs open
    Tseqs = [2.0, 5.0, 10.0, 20.0, 40.0]
    rows = {"T_seq_us": Tseqs, "echo_signal_norm": [], "broken_signal_norm": [], "free_signal_norm": [],
            "echo_Nstar_open": [], "broken_Nstar_open": [], "free_Nstar_open": []}
    for Ts in Tseqs:
        s_echo = np.linalg.norm(first_order_signal(echo_schedule(Ts, 0.5, True), obs))
        s_broken = np.linalg.norm(first_order_signal(echo_schedule(Ts, 0.33, True), obs))
        s_free = np.linalg.norm(first_order_signal(echo_schedule(Ts, 0.5, False), obs))
        rows["echo_signal_norm"].append(float(s_echo)); rows["broken_signal_norm"].append(float(s_broken))
        rows["free_signal_norm"].append(float(s_free))
        # margins use a physical theta; signal scales linearly, so margin = signal_norm * theta_phys
        for key, s in [("echo", s_echo), ("broken", s_broken), ("free", s_free)]:
            Ns = Nstar(s * theta_phys_rate, V_eff)
            rows[f"{key}_Nstar_open"].append(None if not np.isfinite(Ns) else Ns)
    res["sweep"] = rows

    # verdicts vs pre-registration
    echo_blind = all(s < 1e-3 * max(rows["free_signal_norm"]) for s in rows["echo_signal_norm"])
    free_visible = all(s > 1e-2 * max(rows["free_signal_norm"]) for s in rows["free_signal_norm"])
    knob_works = all((rows["broken_Nstar_open"][i] is not None and rows["broken_Nstar_open"][i] < 1e6)
                     for i in range(len(Tseqs)))
    t2_damping = rows["free_signal_norm"][-1] / rows["free_signal_norm"][0]  # signal at 40us / 2us
    res["verdict"] = {
        "(i) K-blind-spot persists under T1/T2": bool(echo_blind),
        "(ii) visible direction survives (margin not collapsed)": bool(free_visible),
        "(ii) T2 damping free-signal(40us)/free-signal(2us)": round(float(t2_damping), 3),
        "(iii) knob still finite-shot-detectable < 1e6 shots": bool(knob_works),
        "V_readout_inflation_factor": round(V_eff, 4),
        "summary": ("PASS if all three hold: echo K-blind under noise; free/broken visible with finite (readout-"
                    "inflated) N*; T2 damps but does not kill the signal at us-scale. Falsifier = a first-order "
                    "signal appearing at the echo (decoherence creating coherent detectability).")}

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.3))
    ax[0].semilogy(Tseqs, np.array(rows["free_signal_norm"]) + 1e-12, "o-", label="free (visible)")
    ax[0].semilogy(Tseqs, np.array(rows["broken_signal_norm"]) + 1e-12, "s-", label="broken echo (knob)")
    ax[0].semilogy(Tseqs, np.array(rows["echo_signal_norm"]) + 1e-12, "^-", label="echo (blind)")
    ax[0].set_xlabel("DD sequence time T_seq (us)"); ax[0].set_ylabel("first-order detuning signal norm (open system)")
    ax[0].legend()
    nb = [n if n else 1e8 for n in rows["broken_Nstar_open"]]
    ne = [n if n else 1e8 for n in rows["echo_Nstar_open"]]
    ax[1].semilogy(Tseqs, nb, "s-", label="broken echo (knob)")
    ax[1].semilogy(Tseqs, ne, "^-", label="echo (blind=INF, plotted 1e8)")
    ax[1].axhline(1e6, ls="--", c="gray", label="1e6 shot budget")
    ax[1].set_xlabel("T_seq (us)"); ax[1].set_ylabel("N* (shots, readout-inflated V)")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_L1_realistic_noise.png"), dpi=120); plt.close(fig)

    with open(os.path.join(OUT, "realistic_noise_results.json"), "w") as f: json.dump(res, f, indent=2, default=str)
    print("\n===== SELF-CHECK L1: realistic open-system noise =====")
    print(f" params: T1={T1_us}us T2={T2_us}us p_ro={P_RO}; V_readout_inflation={V_eff:.4f}")
    print(" T_seq(us):", Tseqs)
    print(" echo signal (blind):  ", [f"{x:.1e}" for x in rows["echo_signal_norm"]])
    print(" broken signal (knob): ", [f"{x:.3f}" for x in rows["broken_signal_norm"]])
    print(" free signal (visible):", [f"{x:.3f}" for x in rows["free_signal_norm"]])
    print(" broken N* (open):     ", [None if n is None else round(n,1) for n in rows["broken_Nstar_open"]])
    print(" VERDICT:", json.dumps(res["verdict"], default=str, indent=2))
    print("=====================================================\n")
    return res

if __name__ == "__main__":
    main()
