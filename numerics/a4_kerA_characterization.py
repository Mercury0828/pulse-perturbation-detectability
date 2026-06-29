"""
A4 characterization: the K-level control-knob blind spots = ker of the time-averaged toggling superoperator
A = (1/T) integral_0^T Ad_{U0(t)^dag} dt.  K_V = T * A(V), so K_V = 0  <=>  V in ker A.

Dichotomy (validated below):
  - generic / free schedules:           ker A = {0}  -> NO K-level blind spot -> control knob INERT.
  - canonical DD (echo, CPMG@T/4,3T/4):  ker A = {Y,Z} (axes anticommuting with the X refocusing) -> knob live.
  - XY4 (universal decoupling):          ker A = {X,Y,Z} -> ALL single-qubit coherent errors K-invisible
                                          (a TOTAL coherent-error blind spot; PDET's knob = modify the sequence
                                          to expose a targeted one).
The blind-spot structure is schedule/timing-SPECIFIC (a non-canonical pulse placement, e.g. CPMG@T/3,2T/3, has
ker A = {0}). This is the honest, case-dependent control-knob characterization -- NOT a universal gamma theorem.

Run: python a4_kerA_characterization.py
"""
import numpy as np
from pdet_core import Schedule, toggling_generator

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]], complex); Z = np.array([[1, 0], [0, -1]], complex)
PAULI = [I, X, Y, Z]; NAMES = ["I", "X", "Y", "Z"]
N, T = 48, 48.0; DT = T / N

def averaging_superoperator(sched):
    """A in the Pauli basis (4x4): A[i,j] = coeff of P_i in (1/T) integral U0^dag P_j U0 dt."""
    Tt = sched.dt * len(sched.H)
    A = np.zeros((4, 4), complex)
    for j, V in enumerate(PAULI):
        K = toggling_generator(sched, [V] * len(sched.H)) / Tt
        for i, P in enumerate(PAULI):
            A[i, j] = np.trace(P.conj().T @ K) / 2
    return A

def sched_from_pulses(positions_axes):
    H = [np.zeros((2, 2), complex) for _ in range(N)]
    for pos, ax in positions_axes:
        H[int(pos * N)] = (np.pi / DT) * ax / 2
    return Schedule(H, DT)

def ker_axes(A, tol=0.05):
    U, sv, Vh = np.linalg.svd(A)
    kdim = int(np.sum(sv < tol))
    axes = [NAMES[i] for k in range(4) if sv[k] < tol for i in range(4) if abs(Vh[k][i]) > 0.7]
    return kdim, axes, [round(float(x), 3) for x in sv]

def main():
    seqs = {
        "free": [], "echo(T/2)": [(0.5, X)],
        "CPMG2(T/4,3T/4)": [(0.25, X), (0.75, X)], "CPMG2(T/3,2T/3)": [(1/3, X), (2/3, X)],
        "XY4": [(0.125, X), (0.375, Y), (0.625, X), (0.875, Y)],
    }
    print(" schedule           dim ker A   blind axes (K-level)        singular values of A")
    for nm, pa in seqs.items():
        A = averaging_superoperator(sched_from_pulses(pa))
        kdim, axes, sv = ker_axes(A)
        print(f" {nm:18s}  {kdim}          {str(axes):26s}  {sv}")
    print("\n ker A characterizes the K-level blind spots; canonical DD (echo/CPMG/XY4) create them, generic does not.")

if __name__ == "__main__":
    main()
