r"""Shared matplotlib style for paper figures.

Figures are authored at their FINAL printed size and saved as vector PDF, so the
LaTeX scale factor is 1.0 and the in-figure type renders at the size set here.

The target is Quantum Science and Technology (IOP Publishing), whose class file
``iopjournal.cls`` sets a single column of

    \textwidth = 153mm = 6.024 in

so a figure included at ``width=\textwidth`` must be authored 6.024 in wide, and
one included at ``width=0.62\textwidth`` must be authored 3.735 in wide. Use
``figsize(frac, height)`` with the SAME fraction the \includegraphics uses --
``check_figure_legibility.py`` reads both back out of the artifacts and fails the
build if they disagree.

Authoring a 12-inch-wide figure and letting \includegraphics squeeze it into a
narrow column scales the type down by the same factor, which is what made the
earlier PNGs illegible. Never set a figsize width by hand.

Type is set at the caption size (8pt, ``\fontsize{8}{10}`` in iopjournal.cls) so
in-figure text matches the caption, and ``pdf.fonttype: 42`` embeds TrueType
rather than Type 3 outlines -- publishers reject PDFs containing Type 3 fonts,
which is the matplotlib default.

The in-figure descriptive titles are intentionally dropped from the paper
figures: the LaTeX \caption carries the description, so an axes title would only
duplicate it (and historically mis-numbered the panels).

Usage:
    import figstyle; figstyle.apply()
    fig, ax = plt.subplots(1, 2, figsize=figstyle.figsize(1.0, 2.5))
    ...
    figstyle.save(fig, OUT, "fig_name")     # writes fig_name.pdf (+ .png preview)
"""
import os
import matplotlib.pyplot as plt

# --- iopjournal.cls page geometry ------------------------------------------
MM_PER_IN = 25.4
TEXT_W = 153.0 / MM_PER_IN   # 6.0236 in -- the full single-column text width

# the fraction used for single-panel figures; keep it in step with the
# \includegraphics[width=...] in the manuscript
SINGLE_PANEL_FRAC = 0.62

CAPTION_PT = 8.0             # iopjournal.cls caption size; in-figure text matches it

PAPER_RC = {
    "font.size": CAPTION_PT,
    "axes.titlesize": CAPTION_PT,
    "axes.labelsize": CAPTION_PT,
    "xtick.labelsize": CAPTION_PT - 0.5,
    "ytick.labelsize": CAPTION_PT - 0.5,
    "legend.fontsize": CAPTION_PT - 0.5,
    "legend.title_fontsize": CAPTION_PT - 0.5,
    "figure.titlesize": CAPTION_PT + 1,
    # line/marker weights scaled for a printed panel a few inches wide
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.2,
    "patch.linewidth": 0.5,
    "grid.linewidth": 0.4,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
    "axes.labelpad": 2.0,
    "legend.frameon": True, "legend.framealpha": 0.85,
    "legend.borderpad": 0.3, "legend.labelspacing": 0.3,
    "legend.handlelength": 1.5, "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.4, "legend.columnspacing": 1.0,
    # never clip a long axis label at the figure edge
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # publishers reject Type 3 fonts; 42 = embed TrueType
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 200,
}

# annotation sizes for call-outs inside a panel (smaller than the axis type)
ANNOT_PT = CAPTION_PT - 1.0


def apply():
    plt.rcParams.update(PAPER_RC)


def figsize(frac=1.0, height=2.2):
    """Final printed size, `frac` being the fraction of \\textwidth the figure occupies.

    Pass the same fraction the manuscript's \\includegraphics asks for: 1.0 for a
    full-width figure, SINGLE_PANEL_FRAC for a single-panel one.
    """
    if not 0.0 < frac <= 1.0:
        raise ValueError("frac must be a fraction of \\textwidth in (0, 1]; "
                         "got %r (the old column-count API is gone)" % (frac,))
    return (frac * TEXT_W, height)


def save(fig, out_dir, stem, png_preview=True):
    """Save the paper-bound vector PDF, plus a raster preview for repo browsing.

    The manuscript includes the .pdf; the .png exists only so the figures render
    on the public repository page.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, stem + ".pdf"))
    if png_preview:
        fig.savefig(os.path.join(out_dir, stem + ".png"), dpi=300)
    plt.close(fig)
