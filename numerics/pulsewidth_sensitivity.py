"""How much of the idle-qubit prediction depends on the assumed pi-pulse width?

The frozen prediction in table `tab:asymxy4` was computed with 2 ns pulses
(`selfcheck_dd_idle_usecase.py` uses pi_dt = dt/50 with dt = T/160 = 100 ns).
The device used for the hardware run, `ibm_cleveland`, has 32 ns X pulses.

That matters asymmetrically, and the distinction decides what the hardware run
can honestly be said to have tested:

  * The asymmetric variant's response is set by the DELIBERATE asymmetry -- the
    toggling integral |int y dt| = T/20 = 0.8 us -- which dwarfs any pulse-width
    residual. Its predicted cost should be essentially pulse-width independent.

  * Symmetric XY4 has no first-order response at all in the ideal-pulse limit, so
    whatever residual it does have is ENTIRELY a finite-pulse effect. Its
    predicted N* is therefore a strong function of the assumed width and the 2 ns
    figure cannot be carried over to a 32 ns device.

Run: python pulsewidth_sensitivity.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import qutip as qt

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "selfcheck")
os.makedirs(OUT, exist_ok=True)

T1_us, T2_us, P_RO = 200.0, 120.0, 0.013
T_SEQ = 16.0
THETA = 0.05
ALPHA = 0.05

sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()
g1 = 1.0 / T1_us
gphi = max(1.0 / T2_us - 1.0 / (2 * T1_us), 0.0)
C_OPS = [np.sqrt(g1) * qt.sigmam(), np.sqrt(gphi / 2) * sz]

SEQS = {
    "xy4": [(0.125, sx), (0.375, sy), (0.625, sx), (0.875, sy)],
    "xy4_asym": [(0.10, sx), (0.375, sy), (0.625, sx), (0.875, sy)],
    "free": [],
}


def evolve(seq, T, drift_rate, rho0, pulse_ns, nfree=160):
    """Same propagation as the frozen self-check, with the pulse width exposed."""
    dt = T / nfree
    pulse_steps = {int(round(f * nfree)): ax for f, ax in SEQS[seq]}
    pi_dt = pulse_ns * 1e-3            # ns -> us
    rho = qt.operator_to_vector(rho0)
    for k in range(nfree):
        rho = (qt.liouvillian((drift_rate / 2.0) * sz, C_OPS) * dt).expm() * rho
        if k in pulse_steps:
            H = (np.pi / pi_dt / 2.0) * pulse_steps[k]
            rho = (qt.liouvillian(H, C_OPS) * pi_dt).expm() * rho
    return qt.vector_to_operator(rho)


def signal(seq, pulse_ns, eps=1e-3):
    rho0 = qt.ket2dm((qt.basis(2, 0) + qt.basis(2, 1)).unit())
    rp = evolve(seq, T_SEQ, +eps, rho0, pulse_ns)
    rm = evolve(seq, T_SEQ, -eps, rho0, pulse_ns)
    return np.array([float((qt.expect(O, rp) - qt.expect(O, rm)) / (2 * eps))
                     for O in (sx, sy, sz)])


def nstar(seq, pulse_ns):
    """Two-point cost at the working point, readout-inflated, as in the self-check."""
    from scipy.stats import norm
    s = np.linalg.norm(signal(seq, pulse_ns)) * THETA
    if s <= 0:
        return float("inf")
    V = 1.0 / (1 - 2 * P_RO) ** 2
    return float((2 * norm.ppf(1 - ALPHA)) ** 2 * V / s ** 2)


def main():
    widths = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    rows = []
    print("\n===== pulse-width sensitivity of the idle-qubit prediction =====")
    print(" T = %.0f us, theta = %.2f rad/us, T1/T2 = %.0f/%.0f us\n" % (T_SEQ, THETA, T1_us, T2_us))
    print(" %10s %14s %14s %14s %12s"
          % ("pulse (ns)", "|ds/dth| XY4", "|ds/dth| asym", "N* XY4", "N* asym"))
    for w in widths:
        s_xy4 = float(np.linalg.norm(signal("xy4", w)))
        s_asy = float(np.linalg.norm(signal("xy4_asym", w)))
        n_xy4, n_asy = nstar("xy4", w), nstar("xy4_asym", w)
        rows.append({"pulse_ns": w, "signal_xy4": s_xy4, "signal_asym": s_asy,
                     "Nstar_xy4": n_xy4, "Nstar_asym": n_asy,
                     "ratio_xy4_over_asym": n_xy4 / n_asy})
        print(" %10.1f %14.3e %14.5f %14.3e %12.0f" % (w, s_xy4, s_asy, n_xy4, n_asy))

    two = next(r for r in rows if r["pulse_ns"] == 2.0)
    dev = next(r for r in rows if r["pulse_ns"] == 32.0)
    print("\n frozen prediction was computed at 2 ns; ibm_cleveland has 32 ns X pulses.")
    print("   asym. XY4 N*:  %.0f (2 ns) -> %.0f (32 ns)   change %.1f%%"
          % (two["Nstar_asym"], dev["Nstar_asym"],
             100 * (dev["Nstar_asym"] / two["Nstar_asym"] - 1)))
    print("   XY4       N*:  %.3e (2 ns) -> %.3e (32 ns)   change %.0fx"
          % (two["Nstar_xy4"], dev["Nstar_xy4"], two["Nstar_xy4"] / dev["Nstar_xy4"]))
    print("\n   => the exposing edit's cost is set by the deliberate asymmetry and is")
    print("      insensitive to the pulse width; the symmetric-XY4 floor is nothing BUT")
    print("      a pulse-width effect, so the 2 ns figure must not be quoted against a")
    print("      32 ns device. Predicted ratio at the device width: %.3e"
          % dev["ratio_xy4_over_asym"])

    res = {"params": {"T_us": T_SEQ, "theta": THETA, "T1_us": T1_us, "T2_us": T2_us,
                      "p_ro": P_RO, "frozen_pulse_ns": 2.0, "device_pulse_ns": 32.0},
           "sweep": rows}
    with open(os.path.join(OUT, "pulsewidth_sensitivity.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("\n wrote results/selfcheck/pulsewidth_sensitivity.json\n")
    return res


if __name__ == "__main__":
    main()
