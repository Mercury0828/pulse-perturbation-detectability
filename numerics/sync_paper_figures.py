"""Copy the manuscript-bound figures from results/ into paper/figures/.

The manuscript includes vector PDFs only; the .png next to each one is a raster
preview for browsing the repository and is deliberately not what LaTeX pulls in.
Two figures (fig_2q_cr, fig_sensitivity) are written straight into paper/figures/
by fig_2q_and_sensitivity.py and so are not listed here.

Run: python sync_paper_figures.py   (also invoked as the last step of run_all.py)
"""
from __future__ import annotations
import os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
PAPER_FIGS = os.path.join(HERE, "..", "paper", "figures")

# (subdirectory of results/, figure stem) -- one entry per \includegraphics in the paper
PAPER_FIGURES = [
    ("phase0",    "fig1_kernel_gamma_1q"),
    ("phase4",    "fig1_invisibility_decomposition"),
    ("phase4",    "fig3_finite_shot_scaling"),
    ("phase4",    "fig7_8_noise_nuisance"),
    ("selfcheck", "fig_L1_realistic_noise"),
    ("selfcheck", "fig_dd_idle_usecase"),
    ("selfcheck", "fig_scalability"),
    ("selfcheck", "fig_model_mismatch"),
]


def main():
    os.makedirs(PAPER_FIGS, exist_ok=True)
    copied, missing = [], []
    for sub, stem in PAPER_FIGURES:
        src = os.path.join(RESULTS, sub, stem + ".pdf")
        if not os.path.exists(src):
            missing.append(os.path.relpath(src, HERE))
            continue
        shutil.copy2(src, os.path.join(PAPER_FIGS, stem + ".pdf"))
        copied.append(stem)
    print("\n===== paper figure sync =====")
    print(" copied %d vector figures into paper/figures/" % len(copied))
    for s in copied:
        print("   %s.pdf" % s)
    if missing:
        print(" MISSING (regenerate the producing module first):")
        for m in missing:
            print("   %s" % m)
    print("=============================\n")
    return {"copied": copied, "missing": missing}


if __name__ == "__main__":
    main()
