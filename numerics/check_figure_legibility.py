"""Verify that in-figure type prints at (at least) the caption size.

The editorial objection that started this check was that figure text was
illegible at ordinary magnification. The cause is mechanical: a figure authored
12 inches wide and pulled into a 6-inch text block has every glyph in it scaled
by 0.5, so nominal 13pt type prints at under 7pt -- and it was far worse under
the earlier two-column target. Font size in the source script is therefore
meaningless on its own -- only size x scale is real.

This script measures the truth from the artifacts themselves: it reads each
figure PDF's MediaBox, reads the width each \\includegraphics asks for out of
the LaTeX source, and reports the resulting scale factor and printed point size.
It also fails the run if any figure is a bitmap or embeds Type 3 fonts, both of
which publishers reject.

Run: python check_figure_legibility.py   (exit code 1 if any figure fails)
"""
from __future__ import annotations
import os, re, sys, glob

import figstyle

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "..", "paper")
SECTIONS = os.path.join(PAPER, "sections")

PT_PER_IN = 72.27          # TeX points
PDF_PT_PER_IN = 72.0       # PostScript points, which is what a PDF MediaBox uses

# a figure is legible if its printed type is at least this size
MIN_PRINTED_PT = 7.5

INCLUDE_RE = re.compile(r"\\includegraphics\[width=([^\]]+)\]\{figures/([^}]+)\}")


def tex_width_inches(spec: str) -> float | None:
    """Resolve a LaTeX width spec such as '0.62\\textwidth' to inches.

    iopjournal.cls is single column, so \\columnwidth == \\textwidth == 153mm.
    """
    spec = spec.strip()
    m = re.match(r"^([\d.]*)\\(columnwidth|textwidth|linewidth)$", spec)
    if not m:
        return None
    frac = float(m.group(1)) if m.group(1) else 1.0
    return frac * figstyle.TEXT_W


def pdf_media_box(path: str):
    """Natural (width, height) of a PDF page in inches, from the first MediaBox."""
    with open(path, "rb") as f:
        blob = f.read()
    m = re.search(rb"/MediaBox\s*\[\s*([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s*\]", blob)
    if not m:
        return None
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    return ((x1 - x0) / PDF_PT_PER_IN, (y1 - y0) / PDF_PT_PER_IN), blob


def main():
    includes = []
    for tex in sorted(glob.glob(os.path.join(SECTIONS, "*.tex"))) + [os.path.join(PAPER, "main.tex")]:
        with open(tex, encoding="utf-8") as f:
            body = f.read()
        for spec, fname in INCLUDE_RE.findall(body):
            includes.append((os.path.basename(tex), spec, fname))

    print("\n===== figure legibility / vector check =====")
    print(" in-figure type is set at %.1fpt (figstyle.CAPTION_PT); printed size = %.1fpt x scale\n"
          % (figstyle.CAPTION_PT, figstyle.CAPTION_PT))
    print(" %-36s %9s %8s %7s %8s  %s" % ("figure", "nat.width", "asked", "scale", "printed", "verdict"))

    bad = []
    for tex, spec, fname in includes:
        path = os.path.join(PAPER, "figures", fname)
        stem = os.path.splitext(fname)[0]
        if not os.path.exists(path):
            print(" %-36s %s" % (stem, "MISSING FILE"))
            bad.append((stem, "missing file"))
            continue
        if not fname.lower().endswith(".pdf"):
            print(" %-36s %s" % (stem, "BITMAP -- a vector PDF is required"))
            bad.append((stem, "bitmap, not vector"))
            continue

        want = tex_width_inches(spec)
        if want is None:
            print(" %-36s unparsed width spec %r" % (stem, spec))
            bad.append((stem, "unparsed width %r" % spec))
            continue

        got = pdf_media_box(path)
        if got is None:
            print(" %-36s %s" % (stem, "no MediaBox"))
            bad.append((stem, "no MediaBox"))
            continue
        (w_in, _h_in), blob = got

        scale = want / w_in
        printed = figstyle.CAPTION_PT * scale
        issues = []
        if printed < MIN_PRINTED_PT:
            issues.append("type prints at %.1fpt" % printed)
        if b"/Type3" in blob:
            issues.append("Type 3 font")
        # a matplotlib "vector" PDF that is really a pasted bitmap has an image XObject
        if b"/Subtype /Image" in blob or b"/Subtype/Image" in blob:
            issues.append("embedded raster image")

        verdict = "ok" if not issues else "FAIL: " + "; ".join(issues)
        if issues:
            bad.append((stem, "; ".join(issues)))
        print(" %-36s %8.2f\" %7.2f\" %7.3f %7.1fpt  %s"
              % (stem, w_in, want, scale, printed, verdict))

    print("\n %d figure inclusions checked, %d problem(s)" % (len(includes), len(bad)))
    for stem, why in bad:
        print("   %-34s %s" % (stem, why))
    print("============================================\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
