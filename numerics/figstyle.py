r"""Shared matplotlib style for paper figures.

Larger fonts for readability in the two-column IEEE layout, and a single place
to keep figure typography uniform. Call figstyle.apply() before plotting.
The in-figure descriptive titles are intentionally dropped from the paper
figures: the LaTeX \caption carries the description, so an axes title would only
duplicate it (and historically mis-numbered the panels).
"""
import matplotlib.pyplot as plt

PAPER_RC = {
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,
    "figure.titlesize": 15,
    # trim whitespace and, crucially, never clip long axis labels at the figure
    # edge (bigger fonts make tight_layout alone insufficient on the wide panels)
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}


def apply():
    plt.rcParams.update(PAPER_RC)
