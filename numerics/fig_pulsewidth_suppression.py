"""Figure: real finite-width pulses make the control-level blind spot a DEEP-BUT-FINITE suppression.
Blind-axis singular value of the averaging superoperator A vs pi-pulse width (echo on 1 qubit). Ideal pulse -> 0;
finite width -> residual ~ width => N* ~ 1/residual^2 large but finite. Genuine A4 finite-pulse data."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pdet_core import Schedule, toggling_generator
X = np.array([[0,1],[1,0]],complex); Z = np.array([[1,0],[0,-1]],complex); Y = np.array([[0,-1j],[1j,0]],complex)
I = np.eye(2,dtype=complex); paulis=[I,X,Y,Z]
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "selfcheck"); os.makedirs(OUT, exist_ok=True)
T = 48.0
def A_blind_sv(width_frac):
    N=200; dt=T/N; nw=max(1,int(width_frac*N))
    H=[np.zeros((2,2),complex) for _ in range(N)]
    for k in range(N//2-nw//2, N//2-nw//2+nw): H[k]=(np.pi/(nw*dt))*X/2
    sc=Schedule(H,dt); Tt=sc.dt*N
    Am=np.zeros((4,4),complex)
    for j,V in enumerate(paulis):
        K=toggling_generator(sc,[V]*N)/Tt
        for i,P in enumerate(paulis): Am[i,j]=np.trace(P.conj().T@K)/2
    return float(np.linalg.svd(Am,compute_uv=False)[-1])
ws=[0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]
sv=[A_blind_sv(w) for w in ws]
supp=[1/s for s in sv]; Nstar=[ (1/s)**2 for s in sv]  # N* ~ 1/sv^2 (relative)
fig,ax=plt.subplots(1,1,figsize=(7,2.9))
ax.loglog(ws, supp, "o-", label="blind-axis suppression $1/\\sigma_{\\min}(A)$")
ax.set_xlabel("$\\pi$-pulse width / sequence time"); ax.set_ylabel("suppression factor")
ax.set_title("Real pulses: deep but finite control-level blind spot")
ax.grid(True, which="both", alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_pulsewidth.png"),dpi=130); plt.close(fig)
json.dump({"width_frac":ws,"blind_sv":sv,"suppression":supp}, open(os.path.join(OUT,"pulsewidth_results.json"),"w"), indent=2)
print("widths:",ws); print("suppression:",[round(s,1) for s in supp])
