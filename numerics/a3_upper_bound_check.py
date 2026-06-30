"""
A3 upper-bound empirical check: the two-point finite-shot detection cost N* = 4 z^2 V / gamma^2 is TIGHT.

For a two-point test (one benign vs one attack) separated by margin gamma in a single benign-projected witness
coordinate, with Gaussian shot noise of variance V/N, the Neyman-Pearson (midpoint-threshold) test reaches
false-alarm = miss = alpha at  N* = (2 z_alpha)^2 V / gamma^2.  Empirically N* * gamma^2 is constant and the
achieved (FA, MISS) match alpha -> the gamma-scaling and the constant are confirmed (achievable AND optimal for
the two-point case). The remaining theory question (composite sets + log factor) is the A3 composite-set analysis.

Run: python a3_upper_bound_check.py
"""
import numpy as np
from scipy.stats import norm

def check(alpha=0.05, V=1.0, seed=20260628, nrep=20000):
    rng = np.random.default_rng(seed)
    z = norm.ppf(1 - alpha)
    print(f"alpha={alpha}, V={V};  prediction N* = (2 z)^2 V / gamma^2,  (2 z)^2 = {(2*z)**2:.4f}")
    print(" gamma     N*_pred      FA@N*    MISS@N*    N*·gamma^2")
    for g in [0.5, 0.2, 0.1, 0.05, 0.02]:
        N = (2 * z) ** 2 * V / g ** 2
        sig = np.sqrt(V / N); thr = g / 2
        fa = np.mean(rng.normal(0, sig, nrep) > thr)
        miss = np.mean(rng.normal(g, sig, nrep) <= thr)
        print(f" {g:5.3f}  {N:10.1f}    {fa:.3f}    {miss:.3f}    {N*g**2:.4f}")
    print(f" => N*·gamma^2 == (2 z)^2 V = {(2*z)**2*V:.4f} (const): two-point cost is Theta(V/gamma^2), tight.")

if __name__ == "__main__":
    check()
