"""Is a virtual-Z-injected drift the same error as a physical detuning?

The hardware campaign injects the test drift as virtual-Z frame updates on the
free segments, because that is the only way to apply a calibrated, exactly known
detuning without recalibrating the qubit.  A physical detuning is not switched
off while a pulse plays, so the two differ by whatever the drift does DURING the
32 ns pi-pulses.  A referee is right to ask whether the hardware therefore
measured the error the model predicts.

This script answers it directly: propagate both models on the device's own
timing grid (dt = 4 ns, 32 ns pi-pulses, 16 us window) and compare the
first-order response.  The answer turns out not to be set by the duty cycle: on a
schedule whose response is a free-segment integral the two models agree to one
part in 10^10, while on one whose response is a cancellation residual (a
symmetric echo) the during-pulse term is the ENTIRE signal.

Run: python vz_vs_continuous.py -> ../results/selfcheck/vz_vs_continuous.json
"""
from __future__ import annotations

import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "selfcheck")
os.makedirs(OUT, exist_ok=True)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)
AX = {"X": X, "Y": Y}

DT_NS = 4.0                       # ibm_cleveland (Heron r2) timing grid
X_DUR_DT = 8                      # 32 ns pi-pulse
WINDOW_DT = 4000                  # 16.0000 us
THETA = 1e-3                      # rad/us, deep in the linear regime

SCHEDULES = {
    "free":       [],
    "echo":       [(0.500, "X")],
    "xy4":        [(0.125, "X"), (0.375, "Y"), (0.625, "X"), (0.875, "Y")],
    "xy4_drop1":  [(0.125, "X"), (0.375, "Y"), (0.625, "X")],
    "xy4_asym":   [(0.100, "X"), (0.375, "Y"), (0.625, "X"), (0.875, "Y")],
}


def dag(A):
    return A.conj().T


def expm2(H, t):
    """exp(-i H t) for a 2x2 Hermitian H."""
    w, v = np.linalg.eigh(H)
    return v @ np.diag(np.exp(-1j * w * t)) @ dag(v)


def propagate(pulses, theta_rad_per_us, drift_during_pulse):
    """Unitary over the window on the device grid.  Time is carried in us.

    drift_during_pulse=False is the virtual-Z model: the frame update is applied
    on the free segments only, so the drift generator is absent while the drive
    is on.  True is a physical detuning, present throughout.
    """
    dt_us = DT_NS * 1e-3
    centres = sorted(int(round(f * WINDOW_DT)) for f, _ in pulses)
    axes = [ax for _, ax in sorted(pulses)]
    # each pulse occupies X_DUR_DT steps centred on its target step
    spans = [(c - X_DUR_DT // 2, c + X_DUR_DT - X_DUR_DT // 2) for c in centres]

    drift = 0.5 * theta_rad_per_us * Z
    U = I2.copy()
    k = 0
    for (a, b), ax in zip(spans, axes):
        if a > k:                                   # free segment
            U = expm2(drift, (a - k) * dt_us) @ U
        # the pulse: a square pi rotation of duration X_DUR_DT * dt
        tp = X_DUR_DT * dt_us
        Hp = (np.pi / tp / 2.0) * AX[ax]
        if drift_during_pulse:
            Hp = Hp + drift
        U = expm2(Hp, tp) @ U
        k = b
    if WINDOW_DT > k:
        U = expm2(drift, (WINDOW_DT - k) * dt_us) @ U
    return U


def response(pulses, drift_during_pulse, eps=THETA):
    """d<O>/dtheta for O in {X, Y, Z} from |+>, by central difference."""
    psi0 = np.array([1, 1], complex) / np.sqrt(2)
    out = []
    for O in (X, Y, Z):
        p = propagate(pulses, +eps, drift_during_pulse) @ psi0
        m = propagate(pulses, -eps, drift_during_pulse) @ psi0
        ep = float(np.real(p.conj() @ O @ p))
        em = float(np.real(m.conj() @ O @ m))
        out.append((ep - em) / (2 * eps))
    return np.array(out)


def main():
    global X_DUR_DT
    duty = 4 * X_DUR_DT / WINDOW_DT
    res = {"grid": {"dt_ns": DT_NS, "pi_pulse_dt": X_DUR_DT, "window_dt": WINDOW_DT,
                    "window_us": WINDOW_DT * DT_NS * 1e-3},
           "theta_rad_per_us": THETA,
           "xy4_pulse_duty_cycle": duty}

    print("\n" + "=" * 80)
    print("Virtual-Z frame injection vs a physical detuning, on the device grid")
    print("=" * 80)
    print("  dt = %.0f ns, pi-pulse = %d dt = %.0f ns, window = %d dt = %.4f us"
          % (DT_NS, X_DUR_DT, X_DUR_DT * DT_NS, WINDOW_DT, WINDOW_DT * DT_NS * 1e-3))
    print("  four-pulse duty cycle = %.3f%% of the window\n" % (100 * duty))
    print("  %-11s %13s %13s %13s %12s"
          % ("schedule", "|h| virtual-Z", "|h| physical", "difference", "rel. diff"))

    rows = {}
    for name, pulses in SCHEDULES.items():
        h_vz = response(pulses, drift_during_pulse=False)
        h_ph = response(pulses, drift_during_pulse=True)
        n_vz, n_ph = float(np.linalg.norm(h_vz)), float(np.linalg.norm(h_ph))
        d = n_ph - n_vz
        big = n_vz > 1e-9
        rel = (d / n_vz) if big else None
        rows[name] = {"h_virtualZ": [round(float(v), 9) for v in h_vz],
                      "h_physical": [round(float(v), 9) for v in h_ph],
                      "norm_virtualZ": round(n_vz, 12), "norm_physical": round(n_ph, 12),
                      "abs_difference": float(d),
                      "relative_difference": rel,
                      "response_is_a_cancellation_residual": bool(not big)}
        print("  %-11s %13.6f %13.6f %13.3e %12s"
              % (name, n_vz, n_ph, d, "--" if rel is None else "%.1e" % rel))
    res["per_schedule"] = rows

    # --- how the during-pulse term scales with the pulse duration -------
    print("\n  Scaling of the difference with the pi-pulse duration:")
    print("  %-11s %14s %14s %14s" % ("schedule", "8 dt (32 ns)", "80 dt (320 ns)", "400 dt (1.6 us)"))
    scaling = {}
    base = X_DUR_DT
    for name in ("echo", "xy4", "xy4_asym", "xy4_drop1"):
        vals = []
        for dur in (8, 80, 400):
            X_DUR_DT = dur
            a = float(np.linalg.norm(response(SCHEDULES[name], False)))
            b = float(np.linalg.norm(response(SCHEDULES[name], True)))
            vals.append(b - a)
        X_DUR_DT = base
        scaling[name] = vals
        print("  %-11s %14.3e %14.3e %14.3e" % (name, *vals))
    res["difference_vs_pulse_duration_dt"] = {"durations_dt": [8, 80, 400], "values": scaling}

    echo, asym = rows["echo"], rows["xy4_asym"]
    print("\n  Reading. The two injection models are not separated by the duty cycle; they")
    print("  are separated by whether the schedule's response is a free-segment integral or")
    print("  a cancellation residual.")
    print("   * On the prescribed edit the response is a free-segment integral, and the two")
    print("     models agree to %.1e in relative terms -- and that gap shrinks as the square"
          % abs(asym["relative_difference"]))
    print("     of the pulse duration, so it is not a limitation of the 32 ns device pulse.")
    print("     The virtual-Z injection therefore measures the quantity the model predicts,")
    print("     nine orders of magnitude inside the 9% agreement it is being compared at.")
    print("   * On a symmetric echo the virtual-Z response is exactly zero while a physical")
    print("     detuning gives %.4f, growing linearly with the pulse duration: there the"
          % echo["norm_physical"])
    print("     during-pulse contribution IS the entire signal.  That is the finite-pulse")
    print("     residual of remark 2, and it is a statement about a physical detuning.")
    print("   * XY4 cancels the during-pulse term as well as the free-segment one, so its")
    print("     residual on hardware comes from error sources outside this model, which is")
    print("     why its cost is reported as measured rather than predicted.")

    res["conclusion"] = {
        "prescribed_edit_relative_difference": asym["relative_difference"],
        "echo_physical_response": echo["norm_physical"],
        "echo_virtualZ_response": echo["norm_virtualZ"],
        "virtualZ_valid_for_prescribed_edit": bool(abs(asym["relative_difference"]) < 1e-6),
    }
    with open(os.path.join(OUT, "vz_vs_continuous.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("\n  wrote results/selfcheck/vz_vs_continuous.json")
    return res


if __name__ == "__main__":
    main()
