"""Build the window-test slide deck as .pptx and a matching .pdf from one spec.

Run:  <venv with python-pptx + matplotlib>/bin/python slides/build_deck.py
The PDF is drawn with matplotlib from the same element list (Carlito =
Calibri metrics), and every text line is checked against its box width.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, Rectangle
from fontTools.ttLib import TTFont
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
FIG = HERE.parent
OUT = HERE / "dms_window_test_slides"
PREVIEW = Path(__import__("os").environ.get("DECK_PREVIEW_DIR", "/tmp"))
W, H = 13.333, 7.5
FONT_PPTX, FONT_PDF = "Calibri", "Carlito"
INK, MUTED = "000000", "595959"
TEAL, ROW1, ROW2 = "1A5A78", "D5DBE1", "E9ECEF"  # table colours (as in the reference deck)
ACCENT = "C00000"  # red emphasis, used sparingly
REGION, WINDOW, BAR = "F2C14E", "2A78D6", "BFBFBF"
MARGIN = 0.08  # text-frame inner margin (in)


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #
@dataclass
class Text:
    x: float
    y: float
    w: float
    h: float
    lines: list[str]  # one entry per paragraph; **bold** markup
    size: float = 18
    align: str = "l"  # l | c
    valign: str = "t"  # t | m
    color: str = INK
    space_after: float = 4  # pt between paragraphs
    border: str | None = None  # border colour -> drawn as a box
    border_pt: float = 2.25
    fill: str | None = None


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float
    fill: str
    line: str | None = None


@dataclass
class Arrow:
    x1: float
    y1: float
    x2: float
    y2: float
    color: str = INK
    pt: float = 2.5


@dataclass
class Img:
    path: Path
    x: float
    y: float
    h: float  # width follows the aspect ratio

    @property
    def w(self) -> float:
        iw, ih = Image.open(self.path).size
        return self.h * iw / ih


@dataclass
class Table:
    x: float
    y: float
    col_w: list[float]
    rows: list[list[str]]  # first row = header
    row_h: float = 0.42
    size: float = 14


@dataclass
class Slide:
    title: str
    items: list = field(default_factory=list)
    notes: str = ""


def runs(s: str) -> list[tuple[str, bool]]:
    parts = re.split(r"\*\*", s)
    return [(p, i % 2 == 1) for i, p in enumerate(parts) if p]


# --------------------------------------------------------------------------- #
# Content
# --------------------------------------------------------------------------- #
def build_spec() -> list[Slide]:
    slides: list[Slide] = []

    # 1 ---- datasets -------------------------------------------------------
    rows = [
        ["Dataset ID", "RNA", "Class", "N (nt)", "Reagent", "R (nt)", "|E|", "Source"],
        ["TRP4P6_DMS_0007", "P4-P6 domain, Tetrahymena ribozyme", "structured", "202", "DMS", "11–51", "22", "RMDB"],
        ["RNASEP_DMS_0000", "RNase P catalytic domain", "structured", "265", "DMS", "4–53", "22", "RMDB"],
        ["HIV3PR_DMS_0001", "HIV-1 3' UTR", "viral UTR", "261", "DMS", "34–83", "20", "RMDB"],
        ["ROTA_RFS11_DMS", "Rotavirus segment 11 (full)", "viral mRNA", "667", "DMS", "6–55", "19", "RASP2"],
        ["HOXA9_STD_0000", "Hox A9 mRNA 5' UTR", "mRNA UTR", "1266", "1M7", "1–100", "100", "RMDB"],
        ["HOXA5_STD_0000", "Hox A5 mRNA 5' UTR", "mRNA UTR", "237", "DMS", "1–50", "31", "RMDB"],
        ["CSDE1_DMS_0000", "Csde1 mRNA, nt 190–386", "mRNA", "197", "DMS", "1–50", "39", "RMDB"],
        ["YXJA_EQR220_BZCN", "B. subtilis yxjA riboswitch leader", "5' leader", "220", "BzCN", "4–53", "50", "RASP2"],
    ]
    s = Slide("Probing datasets: 8 RNAs, all probed in vitro")
    s.items.append(Table(0.5, 1.3, [2.25, 3.55, 1.45, 1.0, 1.05, 1.05, 0.65, 1.33], rows))
    s.items.append(
        Text(
            0.5, 5.4, 6.0, 1.45,
            ["High reactivity → **nucleotide unpaired**",
             "**DMS** reads A, C  ·  **SHAPE** (1M7, BzCN) reads A, C, G, U"],
            size=17, align="c", valign="m", border=INK,
        )
    )
    s.items.append(
        Text(
            6.83, 5.4, 6.0, 1.45,
            ["**R** = first ~50 nt with data at the 5' end",
             "**E** = nucleotides of R that are scored (|E| = 19–100)"],
            size=17, align="c", valign="m", border=INK,
        )
    )
    s.notes = (
        "Six RMDB datasets plus two from RASP v2. R and E are 1-based positions along each RNA. "
        "All conditions, masks and provenance are in the repository README."
    )
    slides.append(s)

    # 2 ---- the test -------------------------------------------------------
    s = Slide("The test: does a window change the prediction?")
    x0, x1 = 0.9, 6.9  # full RNA bar
    s.items += [
        Text(0.5, 1.25, 6.8, 0.4, ["**Full RNA**, length N"], size=16),
        Text(0.35, 1.68, 0.55, 0.4, ["5'"], size=16),
        Text(6.95, 1.68, 0.55, 0.4, ["3'"], size=16),
        Rect(x0, 1.72, x1 - x0, 0.32, BAR),
        Rect(x0, 1.72, 0.85, 0.32, REGION),
        Text(x0, 2.06, 1.6, 0.38, ["region **R**"], size=14),
        Text(0.5, 2.5, 6.8, 0.4, ["**Windows**, anchored at the 5' end"], size=16),
        Rect(x0, 2.95, 1.25, 0.24, WINDOW),
        Text(x0 + 1.32, 2.86, 3.6, 0.4, ["W = smallest window (covers R)"], size=14),
        Rect(x0, 3.33, 3.0, 0.24, WINDOW),
        Text(x0 + 3.07, 3.24, 2.4, 0.4, ["W = 150 nt"], size=14),
        Rect(x0, 3.71, x1 - x0, 0.24, WINDOW),
        Text(x0, 3.97, 6.0, 0.4, ["W = N (reference)   ·   step 10 nt (50 nt for Hox A9)"], size=14, color=MUTED),
        Arrow(3.9, 4.45, 3.9, 4.85),
        Text(
            0.9, 4.9, 6.0, 0.95,
            ["Fold each window: **RNAfold · CONTRAfold · MXfold2**",
             "u(W) = probability that each nucleotide is unpaired"],
            size=16, align="c", valign="m", border=INK,
        ),
        Arrow(3.9, 5.9, 3.9, 6.3),
        Text(0.9, 6.35, 6.0, 0.5, ["Same E for every window and model"], size=16, align="c"),
        # right column: two measures
        Text(
            7.75, 1.3, 5.1, 2.45,
            ["**A. Agreement with data**",
             "ρ(W) = Spearman( u(W), reactivity ) over E",
             "**Δρ(W) = ρ(N) − ρ(W)**",
             "Δρ > 0: window agrees **worse** than full fold"],
            size=17, align="c", valign="m", border=INK, space_after=8,
        ),
        Text(
            7.75, 4.0, 5.1, 2.05,
            ["**B. Change of the model itself**",
             "max over E of |u(W) − u(N)|",
             "converged window: from here on ≤ 0.1",
             "no experimental data used"],
            size=17, align="c", valign="m", border=INK, space_after=8,
        ),
        Text(
            7.75, 6.25, 5.1, 0.75,
            ["95% CI: paired block bootstrap over positions",
             "(2000 resamples, 10-nt blocks)"],
            size=14, align="c", color=MUTED, space_after=0,
        ),
    ]
    s.notes = (
        "u is the unpaired probability from each model's base-pair probabilities. "
        "Spearman is rank-based, so reactivity normalisation does not matter. "
        "A is a data criterion, B is model-internal and can be computed for any sequence."
    )
    slides.append(s)

    # 3 ---- overview summary -------------------------------------------------
    img = Img(FIG / "overview_summary.png", 0.4, 1.3, 5.75)
    s = Slide("Full-length agreement, and the window each RNA needs")
    gx = img.x + img.w + 0.3
    gw = W - 0.45 - gx
    s.items += [
        img,
        Text(gx, 1.3, gw, 2.6,
             ["**(a) ρ(N)**: full fold vs data",
              "bar = 95% CI",
              "**structured: 0.5–0.8**",
              "**mRNA / UTRs: 0.05–0.46**"],
             size=16, align="c", valign="m", border=INK, space_after=6),
        Text(gx, 4.0, gw, 3.0,
             ["**(b) window needed / N**",
              "filled: model converged",
              "open: data criterion met",
              "**almost all ≥ 0.7 N**",
              "except RNAfold on P4-P6",
              "(0.25 N)"],
             size=16, align="c", valign="m", border=INK, space_after=6),
    ]
    s.notes = (
        "Panel (b): W_conv / N (filled) and W*_stable / N (open), the smallest window from which "
        "all larger windows satisfy the criterion. 1.0 means the full length is needed."
    )
    slides.append(s)

    # 4 ---- delta rho --------------------------------------------------------
    img = Img(FIG / "overview_delta_rho.png", 0.4, 1.25, 5.85)
    s = Slide("Cost of truncation depends on the RNA")
    gx = img.x + img.w + 0.35
    gw = W - 0.45 - gx
    s.items += [
        img,
        Text(gx, 1.3, gw, 1.9,
             ["**How to read**",
              "x: window size W (nt)  ·  y: Δρ(W) = ρ(N) − ρ(W)",
              "above 0: window agrees **worse** with data",
              "bands: 95% CI  ·  dashed: ε = 0.02"],
             size=15, align="c", valign="m", border=INK, space_after=4),
        Text(gx, 3.45, gw, 2.4,
             ["**RNase P**: Δρ ≈ 0.3–0.6 at every W < N",
              "**HIV 3' UTR**: fine at 100 nt, jump at 160–190 nt",
              "**Csde1**: up to 0.5 for W ≤ 130 nt",
              "**P4-P6, Hox A5/A9, yxjA**: no clear cost",
              "**Rotavirus**: ~0.2–0.4, very wide CIs"],
             size=16, space_after=7),
        Text(gx, 6.0, gw, 1.0,
             ["Across 8 RNAs: **no significant mean cost**",
              "Wilcoxon p ≥ 0.25 at W = 100 and 150 nt"],
             size=16, align="c", valign="m", border=ACCENT, space_after=2),
    ]
    s.notes = (
        "RNase P: the partner of the 5' helix lies at the 3' end, so any window misses it. "
        "HIV: windows of 160-190 nt cut a helix whose partner lies just outside."
    )
    slides.append(s)

    # 5 ---- convergence ------------------------------------------------------
    img = Img(FIG / "overview_convergence.png", 0.4, 1.25, 5.85)
    s = Slide("The prediction converges only near full length")
    s.items += [
        img,
        Text(gx, 1.3, gw, 1.9,
             ["**How to read**",
              "x: window size W (nt)",
              "y: max over E of |u(W) − u(N)|",
              "dashed: 0.1 (converged)  ·  no data used"],
             size=15, align="c", valign="m", border=INK, space_after=4),
        Text(gx, 3.45, gw, 2.4,
             ["**Converged only at ≥ 0.77 N**, every RNA and model",
              "**RNase P, Hox A9, Csde1, rotavirus**: only at W = N",
              "**P4-P6, Hox A5, HIV**: from 0.77–0.85 N",
              "**RNAfold** moves most: often > 0.8"],
             size=16, space_after=7),
        Text(gx, 6.0, gw, 1.0,
             ["A short window gives a **different fold**",
              "even where Δρ is not significant"],
             size=16, align="c", valign="m", border=ACCENT, space_after=2),
    ]
    s.notes = (
        "W_conv: smallest W from which max |u(W) - u(N)| over E stays <= 0.1 for all larger W. "
        "Values: P4-P6 170-180 of 202, Hox A5 200-230 of 237, HIV 200-240 of 261, yxjA 190-220 of 220."
    )
    slides.append(s)

    # 6 ---- takeaways ----------------------------------------------------------
    s = Slide("Takeaways: which window to use?")
    cw, gap, top = 3.6, 0.52, 1.35
    labels = [
        ("**Fixed window**", "**avoid**: cuts helices, RNA-specific", False),
        ("**Full length**", "**default** (done here up to 1.3 kb)", True),
        ("**Adaptive window**", "**if** the RNA is too long", True),
    ]
    for k, (head, sub, _ok) in enumerate(labels):
        cx = 0.75 + k * (cw + gap)
        s.items.append(Text(cx, top, cw, 0.42, [head], size=18, align="c"))
        s.items.append(Rect(cx + 0.2, top + 0.55, cw - 0.4, 0.26, BAR))
        s.items.append(Rect(cx + 0.2, top + 0.55, 0.5, 0.26, REGION))
        if k == 0:
            s.items.append(Rect(cx + 0.2, top + 0.92, 1.3, 0.2, WINDOW))
        elif k == 1:
            s.items.append(Rect(cx + 0.2, top + 0.92, cw - 0.4, 0.2, WINDOW))
        else:
            for j, frac in enumerate((0.35, 0.6, 0.85)):
                s.items.append(Rect(cx + 0.2, top + 0.92 + j * 0.27, (cw - 0.4) * frac, 0.17, WINDOW))
        s.items.append(Text(cx, top + 1.75, cw, 0.42, [sub], size=16, align="c",
                            color=(ACCENT if k == 0 else INK)))
    s.items.append(
        Text(
            1.2, 3.85, 10.93, 2.25,
            ["**No fixed window is safe**: the needed window depends on each RNA",
             "**Bigger ≠ safer**: a window can cut a helix (HIV, 160–190 nt)",
             "**Policy**: fold the **full length**; if too long, grow W until",
             "max |u(W) − u(N)| ≤ 0.1 over the region, **per RNA**, no data needed"],
            size=19, align="c", valign="m", border=INK, space_after=8,
        )
    )
    s.items.append(
        Text(
            1.2, 6.3, 10.93, 0.75,
            ["Caveats: 8 RNAs, in vitro · 5' regions only (|E| = 19–100) · wide CIs",
             "mRNA UTRs agree weakly with data even at full length"],
            size=14, align="c", color=MUTED, space_after=0,
        )
    )
    s.notes = (
        "Adaptive rule = W_conv: data-free, so it can be applied to any sequence. "
        "In this study it was never less conservative than the data criterion, except RNAfold on Csde1 "
        "where both reached the full length."
    )
    slides.append(s)
    return slides


# --------------------------------------------------------------------------- #
# PPTX writer
# --------------------------------------------------------------------------- #
def rgb(h: str) -> RGBColor:
    return RGBColor.from_string(h)


def _fill_tf(tf, item: Text) -> None:
    tf.word_wrap = True
    for side in ("left", "right"):
        setattr(tf, f"margin_{side}", Inches(MARGIN))
    tf.margin_top = tf.margin_bottom = Inches(0.04)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE if item.valign == "m" else MSO_ANCHOR.TOP
    for i, line in enumerate(item.lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER if item.align == "c" else PP_ALIGN.LEFT
        if i < len(item.lines) - 1:
            p.space_after = Pt(item.space_after)
        for txt, bold in runs(line):
            r = p.add_run()
            r.text = txt
            r.font.size = Pt(item.size)
            r.font.bold = bold
            r.font.name = FONT_PPTX
            r.font.color.rgb = rgb(item.color)


def write_pptx(slides: list[Slide], path: Path) -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    blank = prs.slide_layouts[6]
    for n, sd in enumerate(slides, start=1):
        sl = prs.slides.add_slide(blank)
        t = Text(0.45, 0.3, W - 0.9, 0.85, [f"**{sd.title}**"], size=36, valign="m")
        tb = sl.shapes.add_textbox(Inches(t.x), Inches(t.y), Inches(t.w), Inches(t.h))
        tb.name = "Title"
        _fill_tf(tb.text_frame, t)
        for it in sd.items:
            if isinstance(it, Text):
                if it.border or it.fill:
                    shp = sl.shapes.add_shape(
                        MSO_SHAPE.RECTANGLE, Inches(it.x), Inches(it.y), Inches(it.w), Inches(it.h)
                    )
                    if it.fill:
                        shp.fill.solid()
                        shp.fill.fore_color.rgb = rgb(it.fill)
                    else:
                        shp.fill.background()
                    shp.line.color.rgb = rgb(it.border or it.fill or INK)
                    shp.line.width = Pt(it.border_pt)
                    shp.shadow.inherit = False
                    _fill_tf(shp.text_frame, it)
                else:
                    tb = sl.shapes.add_textbox(Inches(it.x), Inches(it.y), Inches(it.w), Inches(it.h))
                    _fill_tf(tb.text_frame, it)
            elif isinstance(it, Rect):
                shp = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(it.x), Inches(it.y), Inches(it.w), Inches(it.h))
                shp.fill.solid()
                shp.fill.fore_color.rgb = rgb(it.fill)
                if it.line:
                    shp.line.color.rgb = rgb(it.line)
                else:
                    shp.line.fill.background()
                shp.shadow.inherit = False
            elif isinstance(it, Arrow):
                c = sl.shapes.add_connector(
                    MSO_CONNECTOR.STRAIGHT, Inches(it.x1), Inches(it.y1), Inches(it.x2), Inches(it.y2)
                )
                c.line.color.rgb = rgb(it.color)
                c.line.width = Pt(it.pt)
                ln = c.line._get_or_add_ln()
                tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
                ln.append(tail)
            elif isinstance(it, Img):
                pic = sl.shapes.add_picture(str(it.path), Inches(it.x), Inches(it.y), height=Inches(it.h))
                pic.name = it.path.stem
                pic._element.nvPicPr.cNvPr.set("descr", it.path.stem.replace("_", " "))
            elif isinstance(it, Table):
                nr, nc = len(it.rows), len(it.col_w)
                gt = sl.shapes.add_table(
                    nr, nc, Inches(it.x), Inches(it.y), Inches(sum(it.col_w)), Inches(it.row_h * nr)
                ).table
                for j, cw in enumerate(it.col_w):
                    gt.columns[j].width = Inches(cw)
                for i, row in enumerate(it.rows):
                    gt.rows[i].height = Inches(it.row_h)
                    for j, val in enumerate(row):
                        cell = gt.cell(i, j)
                        cell.fill.solid()
                        cell.fill.fore_color.rgb = rgb(TEAL if i == 0 else (ROW1 if i % 2 else ROW2))
                        cell.margin_left = cell.margin_right = Inches(MARGIN)
                        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                        p = cell.text_frame.paragraphs[0]
                        r = p.add_run()
                        r.text = val
                        r.font.size = Pt(it.size)
                        r.font.name = FONT_PPTX
                        r.font.bold = i == 0
                        r.font.color.rgb = rgb("FFFFFF" if i == 0 else INK)
        num = sl.shapes.add_textbox(Inches(W - 0.9), Inches(H - 0.45), Inches(0.5), Inches(0.3))
        _fill_tf(num.text_frame, Text(0, 0, 0, 0, [str(n)], size=11, color=MUTED))
        num.text_frame.paragraphs[0].alignment = PP_ALIGN.RIGHT
        if sd.notes:
            sl.notes_slide.notes_text_frame.text = sd.notes
    prs.save(path)


# --------------------------------------------------------------------------- #
# PDF renderer (+ overflow check)
# --------------------------------------------------------------------------- #
DPI = 150
problems: list[str] = []


_FONTS = {
    False: TTFont("/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf"),
    True: TTFont("/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf"),
}


def advance(txt: str, bold: bool, size: float) -> float:
    """Text advance width in inches from the font's glyph advances (Carlito = Calibri metrics)."""
    f = _FONTS[bold]
    cmap, hmtx, upem = f.getBestCmap(), f["hmtx"], f["head"].unitsPerEm
    units = 0
    for ch in txt:
        g = cmap.get(ord(ch))
        if g is None:
            problems.append(f"glyph missing in Carlito: {ch!r}")
            g = cmap[ord("x")]
        units += hmtx[g][0]
    return units / upem * size / 72


def _hex(h: str) -> str:
    return "#" + h


def _draw_text(ax, fig, it: Text, where: str) -> None:
    lh = it.size * 1.2 / 72
    heights = [lh] * len(it.lines)
    total = sum(heights) + it.space_after / 72 * (len(it.lines) - 1)
    inner_h = it.h - 0.08
    if total > inner_h + 1e-3:
        problems.append(f"{where}: text height {total:.2f} > box {inner_h:.2f} ({it.lines[0][:40]!r})")
    y = it.y + 0.04 + (inner_h - total) / 2 if it.valign == "m" else it.y + 0.04
    for line in it.lines:
        parts = []
        for txt, bold in runs(line):
            t = ax.text(0, 0, txt, fontsize=it.size, fontfamily=FONT_PDF,
                        fontweight="bold" if bold else "normal", color=_hex(it.color), va="top")
            # drawn width of the visible text + font advance of leading/trailing spaces
            # (matplotlib's extent ignores surrounding spaces)
            core = txt.strip(" ")
            lead = len(txt) - len(txt.lstrip(" "))
            trail = len(txt) - len(txt.rstrip(" "))
            t.set_text(core)
            w_core = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / DPI if core else 0.0
            t.set_text(txt)
            parts.append((t, w_core + advance(" " * (lead + trail), bold, it.size), advance(" " * lead, bold, it.size)))
        width = sum(w for _, w, _ in parts)
        avail = it.w - 2 * MARGIN
        if width > avail + 1e-3:
            problems.append(f"{where}: line {width:.2f} in > {avail:.2f} in: {line!r}")
        x = it.x + MARGIN + ((avail - width) / 2 if it.align == "c" else 0)
        for t, w, lead in parts:
            t.set_text(t.get_text().lstrip(" "))
            t.set_position((x + lead, y + 0.02))
            x += w
        y += lh + it.space_after / 72


def render_pdf(slides: list[Slide], path: Path) -> None:
    with PdfPages(path) as pdf:
        for n, sd in enumerate(slides, start=1):
            fig = plt.figure(figsize=(W, H), dpi=DPI)
            ax = fig.add_axes((0, 0, 1, 1))
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.axis("off")
            where = f"slide {n}"
            _draw_text(ax, fig, Text(0.45, 0.3, W - 0.9, 0.85, [f"**{sd.title}**"], size=36, valign="m"), where)
            for it in sd.items:
                if isinstance(it, Text):
                    if it.border or it.fill:
                        ax.add_patch(Rectangle((it.x, it.y), it.w, it.h,
                                               facecolor=_hex(it.fill) if it.fill else "none",
                                               edgecolor=_hex(it.border or it.fill or INK),
                                               linewidth=it.border_pt))
                    _draw_text(ax, fig, it, where)
                elif isinstance(it, Rect):
                    ax.add_patch(Rectangle((it.x, it.y), it.w, it.h, facecolor=_hex(it.fill),
                                           edgecolor=_hex(it.line) if it.line else "none"))
                elif isinstance(it, Arrow):
                    ax.add_patch(FancyArrowPatch((it.x1, it.y1), (it.x2, it.y2), arrowstyle="-|>",
                                                 mutation_scale=18, linewidth=it.pt, color=_hex(it.color)))
                elif isinstance(it, Img):
                    ax.imshow(Image.open(it.path), extent=(it.x, it.x + it.w, it.y + it.h, it.y),
                              interpolation="lanczos")
                    if it.x + it.w > W - 0.3 or it.y + it.h > H - 0.2:
                        problems.append(f"{where}: image {it.path.name} runs off the slide")
                elif isinstance(it, Table):
                    y = it.y
                    for i, row in enumerate(it.rows):
                        x = it.x
                        for j, val in enumerate(row):
                            cw = it.col_w[j]
                            fill = TEAL if i == 0 else (ROW1 if i % 2 else ROW2)
                            ax.add_patch(Rectangle((x, y), cw, it.row_h, facecolor=_hex(fill),
                                                   edgecolor="white", linewidth=1))
                            cell = Text(x, y + (it.row_h - it.size * 1.2 / 72 - 0.08) / 2, cw,
                                        it.size * 1.2 / 72 + 0.08,
                                        [f"**{val}**" if i == 0 else val], size=it.size,
                                        color="FFFFFF" if i == 0 else INK)
                            _draw_text(ax, fig, cell, f"{where} table")
                            x += cw
                        y += it.row_h
                    if sum(it.col_w) + it.x > W - 0.4:
                        problems.append(f"{where}: table wider than slide")
            _draw_text(ax, fig, Text(W - 0.9, H - 0.45, 0.5, 0.3, [str(n)], size=11, color=MUTED, align="c"), where)
            pdf.savefig(fig)
            fig.savefig(PREVIEW / f"preview_{n}.png", dpi=110)
            plt.close(fig)


def main() -> None:
    slides = build_spec()
    write_pptx(slides, OUT.with_suffix(".pptx"))
    render_pdf(slides, OUT.with_suffix(".pdf"))
    print("\n".join(problems) if problems else "No overflow detected.")


if __name__ == "__main__":
    main()
