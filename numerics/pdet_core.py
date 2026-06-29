"""
PDET Phase-0 numerics — core response-map machinery.

Implements (guide §4):
  - piecewise-constant pulse propagators U0(t_k)   (scipy.linalg.expm; QuTiP cross-check elsewhere)
  - toggling-frame generators  K_j = ∫ U0(t)† V_j(t) U0(t) dt
  - first-order restricted response map  M_{(l,s),j} = -i Tr[[rho_s, Õ_l] K_j],  Õ_l = U0(T)† O_l U0(T)
  - first-order-invisible subspace  ker M  (right null space; via SVD)
  - operational margin  gamma = sigma_min( P_B^perp M )   (worst-case, benign-projected)
  - second-order margin eta2  (exact central finite difference on kernel directions)

CORRECTNESS ANCHOR (A1, failure-mode #1): a built-in finite-difference check verifies the closed-form M against
exact propagation of U_{eps*theta}. No gamma/kernel number is trusted unless A1 passes. Run `python pdet_core.py`.

NB on the interaction-frame observable: the exact lab-frame first-order signal is
  d/d eps Tr[O_l U_{eps theta} rho_s U_{eps theta}^†]|_0 = -i Tr[[rho_s, Õ_l] L_theta],  Õ_l = U0(T)† O_l U0(T).
This is the guide's -i Tr[[rho_s,O_l] L_theta] with O in the toggling frame. We implement the exact version.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm

# ----------------------------------------------------------------------------- operators
def qubit_ops():
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z

def qutrit_ops():
    # 3-level transmon: annihilation a, number n, identity
    a = np.zeros((3, 3), dtype=complex)
    a[0, 1] = 1.0
    a[1, 2] = np.sqrt(2.0)
    ad = a.conj().T
    n = ad @ a
    I3 = np.eye(3, dtype=complex)
    return I3, a, ad, n

def dag(A): return A.conj().T
def comm(A, B): return A @ B - B @ A
def ketbra(i, j, d):
    M = np.zeros((d, d), dtype=complex); M[i, j] = 1.0; return M

# ----------------------------------------------------------------------------- schedule / propagators
class Schedule:
    """Piecewise-constant control. step_hams: list of (H_drift+H_ctrl(t_k)) d×d arrays; dt: step length (ns)."""
    def __init__(self, step_hams, dt):
        self.H = [np.asarray(h, dtype=complex) for h in step_hams]
        self.dt = float(dt)
        self.d = self.H[0].shape[0]
        self._cum = None

    def step_props(self):
        return [expm(-1j * h * self.dt) for h in self.H]

    def cumulative_props(self):
        """U0(t_k) = propagator from 0 to the END of step k.  Returns list length nsteps; plus U0(T)."""
        if self._cum is not None:
            return self._cum
        sp = self.step_props()
        U = np.eye(self.d, dtype=complex)
        cum = []
        for s in sp:
            U = s @ U
            cum.append(U.copy())
        self._cum = cum
        return cum

    def U0(self):
        return self.cumulative_props()[-1]

    def U0_at_midpoints(self):
        """U0(t) evaluated at step midpoints (for the toggling-frame integral)."""
        sp = self.step_props()
        half = [expm(-1j * h * self.dt / 2.0) for h in self.H]
        U = np.eye(self.d, dtype=complex)
        mids = []
        for k, s in enumerate(sp):
            mids.append(half[k] @ U)   # propagate to midpoint of step k
            U = s @ U
        return mids

# ----------------------------------------------------------------------------- toggling-frame K_j
def _within_step_integral(Hk, Vk, dt):
    """
    EXACT G_k = int_0^dt e^{i Hk tau} Vk e^{-i Hk tau} dtau  (Hk piecewise-constant on the step).
    In the Hk eigenbasis: (G_k)_{ab} = Vk_{ab} * phi(lam_a - lam_b), phi(w) = (e^{i w dt}-1)/(i w), phi(0)=dt.
    This makes the toggling generator EXACT for the discrete piecewise-constant propagation (A1 -> machine eps).
    """
    lam, Q = np.linalg.eigh(Hk)
    Vtil = dag(Q) @ Vk @ Q
    dw = lam[:, None] - lam[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = (np.exp(1j * dw * dt) - 1.0) / (1j * dw)
    phi[np.abs(dw) < 1e-12] = dt
    G = Q @ (Vtil * phi) @ dag(Q)
    return G

def toggling_generator(sched: Schedule, V_steps, exact=True):
    """
    K = int_0^T U0(t)^dag V(t) U0(t) dt.  exact=True uses the exact within-step conjugation integral
    (machine-precision consistent with the discrete propagation); exact=False uses the midpoint rule.
    L = sum_k P_{k-1}^dag G_k P_{k-1},  P_{k-1} = U0 at the START of step k.
    """
    d = sched.d
    K = np.zeros((d, d), dtype=complex)
    if exact:
        sp = sched.step_props()
        P = np.eye(d, dtype=complex)            # P_{k-1}, propagator to start of step k
        for k in range(len(sched.H)):
            Gk = _within_step_integral(sched.H[k], np.asarray(V_steps[k], dtype=complex), sched.dt)
            K += dag(P) @ Gk @ P
            P = sp[k] @ P
        return K
    mids = sched.U0_at_midpoints()
    for k, U in enumerate(mids):
        K += dag(U) @ np.asarray(V_steps[k], dtype=complex) @ U * sched.dt
    return K

# ----------------------------------------------------------------------------- response map M
def response_map(sched: Schedule, K_list, states, obs):
    """
    M_{(l,s), j} = -i Tr[[rho_s, Õ_l] K_j],  Õ_l = U0(T)† O_l U0(T).
    states: list of rho_s (d×d). obs: list of O_l (d×d). K_list: list of K_j.
    Returns real matrix M of shape (len(obs)*len(states), len(K_list)).
    """
    U0T = sched.U0()
    Otil = [dag(U0T) @ O @ U0T for O in obs]
    rows = []
    for O in Otil:
        for rho in states:
            C = comm(rho, O)              # [rho_s, Õ_l]
            row = [(-1j * np.trace(C @ K)) for K in K_list]
            rows.append(row)
    M = np.array(rows, dtype=complex)
    # physical first-order signal is real; imaginary part is numerical noise — assert + drop.
    # Use BOTH a relative and an absolute floor: a genuinely invisible (blind) direction has M~0, so a relative-
    # only check false-triggers on sub-1e-15 numerical noise when there is no real signal to compare against.
    imag = np.max(np.abs(M.imag)) if M.size else 0.0
    if imag > 1e-9 * (np.max(np.abs(M.real)) + 1e-15) and imag > 1e-11:
        raise RuntimeError(f"response_map: unexpected imaginary part {imag:.2e} (check Hermiticity of V/K).")
    return M.real

# ----------------------------------------------------------------------------- kernel / rank / gamma
def singular_spectrum(M):
    if M.size == 0:
        return np.array([])
    return np.linalg.svd(M, compute_uv=False)

def kernel_dim(M, rtol=1e-9):
    s = singular_spectrum(M)
    if s.size == 0:
        return M.shape[1]
    smax = s[0]
    rank = int(np.sum(s > rtol * smax))
    return M.shape[1] - rank

def kernel_basis(M, rtol=1e-9):
    """Right null space (theta directions with zero first-order signal)."""
    U, s, Vh = np.linalg.svd(M)
    smax = s[0] if s.size else 0.0
    rank = int(np.sum(s > rtol * smax)) if s.size else 0
    return Vh[rank:].conj().T   # columns = kernel basis vectors (in theta space)

def benign_projector(M, benign_idx):
    """P_B^perp onto signal space orthogonal to span{ M e_b : b in benign_idx }."""
    nrows = M.shape[0]
    if not benign_idx:
        return np.eye(nrows)
    B = M[:, benign_idx]                  # signal-space spans of benign directions
    Ub, sb, _ = np.linalg.svd(B, full_matrices=False)
    keep = Ub[:, sb > 1e-12 * (sb[0] if sb.size else 1.0)]
    P = np.eye(nrows) - keep @ keep.conj().T
    return P

def gamma_margin(M, benign_idx=None, attack_cols=None):
    """
    gamma = sigma_min( P_B^perp M[:, attack_cols] ) — smallest accessible signal of a unit worst-case (kernel-
    nearest) attack after projecting out the benign subspace. attack_cols=None → all directions.
    Returns (gamma, full singular spectrum of P_B^perp M_A).
    """
    benign_idx = benign_idx or []
    P = benign_projector(M, benign_idx)
    MA = M if attack_cols is None else M[:, attack_cols]
    PM = P @ MA
    s = singular_spectrum(PM)
    gamma = float(s[-1]) if s.size and PM.shape[1] <= PM.shape[0] else (float(s[-1]) if s.size else 0.0)
    return gamma, s

# ----------------------------------------------------------------------------- exact perturbed signal (for A1 + eta2)
def perturbed_signal(sched_builder, theta, dictionary_Vsteps, states, obs):
    """
    Exact g(theta)_{l,s} = Tr[O_l U_theta rho_s U_theta†], where U_theta is the EXACT propagator of
    H0(t) + sum_j theta_j V_j(t) (piecewise constant). sched_builder() returns (step_hams_list, dt).
    """
    step_hams, dt = sched_builder()
    d = step_hams[0].shape[0]
    nsteps = len(step_hams)
    U = np.eye(d, dtype=complex)
    for k in range(nsteps):
        Hk = np.asarray(step_hams[k], dtype=complex).copy()
        for j, Vj in enumerate(dictionary_Vsteps):
            Hk = Hk + theta[j] * np.asarray(Vj[k], dtype=complex)
        U = expm(-1j * Hk * dt) @ U
    g = np.zeros((len(obs), len(states)))
    for li, O in enumerate(obs):
        for si, rho in enumerate(states):
            val = np.trace(O @ U @ rho @ dag(U))
            g[li, si] = val.real
    return g  # shape (L, S)

def a1_finite_diff_check(sched_builder, sched: Schedule, dictionary_Vsteps, states, obs, eps=1e-6, ntest=5, seed=0):
    """
    A1 CORRECTNESS ANCHOR. Compare closed-form M·theta to the exact central finite-difference derivative of the
    perturbed signal, for random theta. Returns max relative error. Must be << 1 (we require < 1e-4).
    """
    rng = np.random.default_rng(seed)
    K_list = [toggling_generator(sched, V) for V in dictionary_Vsteps]
    M = response_map(sched, K_list, states, obs)        # (L*S, m)
    L, S = len(obs), len(states)
    max_rel = 0.0
    for _ in range(ntest):
        theta = rng.standard_normal(len(dictionary_Vsteps))
        theta /= np.linalg.norm(theta)
        gp = perturbed_signal(sched_builder, eps * theta, dictionary_Vsteps, states, obs)
        gm = perturbed_signal(sched_builder, -eps * theta, dictionary_Vsteps, states, obs)
        deriv = (gp - gm) / (2 * eps)                   # exact d g / d eps at theta  → shape (L,S)
        # closed form: (M theta) reshaped to (L,S) matching row order (l outer, s inner)
        pred = (M @ theta).reshape(L, S)
        denom = np.max(np.abs(deriv)) + 1e-12
        max_rel = max(max_rel, np.max(np.abs(deriv - pred)) / denom)
    return max_rel, M

def eta2_for_direction(sched_builder, theta, dictionary_Vsteps, states, obs, benign_P=None, eps=1e-3):
    """
    Second-order accessible signal magnitude for a (first-order-invisible) direction theta.
    Central 2nd difference: Q ≈ (g(eps)+g(-eps)-2 g(0))/eps^2, flattened; eta2 = ||P_B^perp vec(Q)||.
    """
    g0 = perturbed_signal(sched_builder, 0 * theta, dictionary_Vsteps, states, obs)
    gp = perturbed_signal(sched_builder, eps * theta, dictionary_Vsteps, states, obs)
    gm = perturbed_signal(sched_builder, -eps * theta, dictionary_Vsteps, states, obs)
    Q = (gp + gm - 2 * g0) / (eps ** 2)
    v = Q.reshape(-1)
    if benign_P is not None:
        v = benign_P @ v
    return float(np.linalg.norm(v))

# ----------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    I, X, Y, Z = qubit_ops()
    print("=== PDET core self-test ===")

    # (1) NEGATIVE CONTROL: 1 qubit, free-ish evolution, S={|0>}, O={X}; detuning dir (K∝Z) must be invisible.
    d = 2
    nsteps, dt = 20, 1.0
    # near-identity drift (tiny) so U0≈I; detuning V=Z
    H0 = [0.001 * Z for _ in range(nsteps)]
    sched = Schedule(H0, dt)
    Vdet = [Z for _ in range(nsteps)]           # detuning direction
    Vx   = [X for _ in range(nsteps)]           # an X-drive error direction
    Kdet = toggling_generator(sched, Vdet)
    Kx   = toggling_generator(sched, Vx)
    rho0 = ketbra(0, 0, d)
    M = response_map(sched, [Kdet, Kx], [rho0], [X])
    print(f" neg-control M (S={{|0>}}, O={{X}}):\n  M = {np.round(M,5).tolist()}")
    print(f"  detuning(Z) column |M[:,0]| = {np.abs(M[:,0]).max():.2e}  (expect ~0 -> invisible)")
    print(f"  Xerror   column |M[:,1]| = {np.abs(M[:,1]).max():.2e}")
    print(f"  dim ker M = {kernel_dim(M)}  (expect >=1)")

    # (2) A1 finite-difference check on a real DRAG-ish 1q schedule with a 4-direction dictionary.
    rng = np.random.default_rng(1)
    # random-but-fixed piecewise control as a stand-in schedule (full DRAG model lives in models.py)
    Hsteps = [0.5 * rng.standard_normal() * X + 0.5 * rng.standard_normal() * Y + 0.2 * Z for _ in range(nsteps)]
    sched2 = Schedule(Hsteps, dt)
    dict_V = [[X]*nsteps, [Y]*nsteps, [Z]*nsteps, [(X+Z)/np.sqrt(2)]*nsteps]
    states = [ketbra(0,0,d), (ketbra(0,0,d)+ketbra(1,1,d)+ketbra(0,1,d)+ketbra(1,0,d))/2]  # |0>, |+>
    obs = [Z, X]
    def builder():
        return ([h.copy() for h in Hsteps], dt)
    max_rel, M2 = a1_finite_diff_check(builder, sched2, dict_V, states, obs, eps=1e-6, ntest=6, seed=3)
    print(f"\n A1 finite-difference max relative error = {max_rel:.2e}  (require < 1e-4)")
    print(f"  M2 shape {M2.shape}, singular spectrum = {np.round(singular_spectrum(M2),4).tolist()}")
    print(f"  dim ker M2 = {kernel_dim(M2)}")
    assert max_rel < 1e-4, "A1 CHECK FAILED - response map disagrees with exact propagation."
    print("\n A1 PASSED. Core response map validated against exact propagation.")
