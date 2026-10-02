"""Generate the IEEE 13-node single-line diagram (schematic, not exact geometry).

Draws a simplified single-line diagram (SLD) of this project's modified
IEEE-13 reconstruction (see src/model/ieee13_data.py and
experiments/exp02_13bus.py) in the same dark, color-coded visual style as
report/figures/ieee4.jpg, and saves it to report/figures/ieee13.jpg (plus a
matching .pdf).

This is intentionally a SCHEMATIC simplification, not an exact reproduction
of the official IEEE-13 branch geometry:
  - Bus 670 (a pure pass-through node on line 632-671, no load) is folded
    into a single drawn segment "632 -> 671".
  - Bus 680 (a dead-end switch node off 671 with no load) is omitted.
  - Buses 611 and 652 are drawn as two leaves off a single junction bus 684,
    matching the real topology (671 -> 684 -> {611, 652}) without needing
    additional intermediate geometry.
All OTHER real structure is kept: the full backbone (SourceBus -> T1 -> 650
-> 632 -> 671), three real laterals off the backbone (632 -> 633 -> T2 ->
634; 632 -> 645 -> 646; 671 -> 675; 671 -> 684 -> {611, 652}), both
paper-specified transformers with their actual winding connections (T1:
Yg-Delta, T2: Delta-Delta), and the project's three voltage zones (115 kV
source zone, 4.16 kV primary-distribution zone, 0.48 kV secondary zone at
bus 634) exactly as documented in experiments/exp02_13bus.py's module
docstring and docs/project_guide.tex.

Run from the repository root: ``python report/make_ieee13_diagram.py``.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Palette (matched to report/figures/ieee4.jpg)
# ---------------------------------------------------------------------------
BG = "#0d1526"
FG_TITLE = "#f2f4f8"
FG_SUB = "#9aa4b6"
GREEN = "#57d68d"
ORANGE = "#e8a33d"
PURPLE = "#8f7ee8"
PURPLE_FILL = "#241c3d"
RED = "#e8615c"
RED_FILL = "#2c1416"
BLUE = "#5b8def"
GRAY_LINE = "#6b7484"
GRAY_TEXT = "#aab2c2"
ZONE_115_EDGE = "#5b8def"
ZONE_115_FILL = "#16223f"
ZONE_416_EDGE = "#57d68d"
ZONE_416_FILL = "#12241f"
ZONE_048_EDGE = "#c77dea"
ZONE_048_FILL = "#241c33"

FONT = "DejaVu Sans Mono"

fig, ax = plt.subplots(figsize=(15, 9), dpi=180)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 15)
ax.set_ylim(0, 9)
ax.axis("off")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def bus_tick(x, y, label, color=BLUE, label_pos="below"):
    ax.plot([x, x], [y - 0.22, y + 0.22], color=color, linewidth=3.2, solid_capstyle="round", zorder=6)
    if label_pos == "below":
        ax.text(x, y - 0.40, label, ha="center", va="top", color=color, fontsize=10,
                fontweight="bold", family=FONT, zorder=6)
    elif label_pos == "right":
        ax.text(x + 0.30, y, label, ha="left", va="center", color=color, fontsize=10,
                fontweight="bold", family=FONT, zorder=6)
    elif label_pos == "left":
        ax.text(x - 0.30, y, label, ha="right", va="center", color=color, fontsize=10,
                fontweight="bold", family=FONT, zorder=6)


def ground(x, y):
    widths = [0.16, 0.11, 0.06]
    for i, w in enumerate(widths):
        yy = y - i * 0.10
        ax.plot([x - w, x + w], [yy, yy], color=GRAY_LINE, linewidth=1.6, zorder=4)


def hline(x0, x1, y, color=GRAY_LINE, lw=1.8):
    ax.plot([x0, x1], [y, y], color=color, linewidth=lw, zorder=2)


def vline(x, y0, y1, color=GRAY_LINE, lw=1.8):
    ax.plot([x, x], [y0, y1], color=color, linewidth=lw, zorder=2)


def h_label(x0, x1, y, text, boxed=True, fontsize=8.3):
    xm = (x0 + x1) / 2
    if boxed:
        w = max(1.05, 0.11 * len(text.split(chr(10))[0]) + 0.35)
        box = FancyBboxPatch((xm - w / 2, y - 0.20), w, 0.40,
                              boxstyle="round,pad=0.02,rounding_size=0.06",
                              linewidth=1.3, edgecolor=PURPLE, facecolor=PURPLE_FILL, zorder=4)
        ax.add_patch(box)
        ax.text(xm, y, text, ha="center", va="center", color=PURPLE, fontsize=fontsize,
                fontweight="bold", family=FONT, zorder=5)
    else:
        ax.text(xm, y + 0.26, text, ha="center", va="bottom", color=GRAY_TEXT, fontsize=7.2,
                family=FONT, zorder=5)


def v_label(x, y0, y1, text, side=1, fontsize=7.2):
    ym = (y0 + y1) / 2
    ax.text(x + side * 0.30, ym, text, ha="left" if side > 0 else "right", va="center",
            color=GRAY_TEXT, fontsize=fontsize, family=FONT, zorder=5)


def transformer(x, y, label, sublabel):
    ax.add_patch(Circle((x - 0.17, y), 0.32, facecolor="none", edgecolor=ORANGE, linewidth=2.0, zorder=5))
    ax.add_patch(Circle((x + 0.17, y), 0.32, facecolor="none", edgecolor=ORANGE, linewidth=2.0, zorder=5))
    ax.text(x, y + 0.56, "T", ha="center", va="bottom", color=ORANGE, fontsize=11,
            fontweight="bold", family=FONT, zorder=6)
    ax.text(x, y - 0.50, label, ha="center", va="top", color=ORANGE, fontsize=8.6,
            fontweight="bold", family=FONT, zorder=6)
    ax.text(x, y - 0.75, sublabel, ha="center", va="top", color=ORANGE, fontsize=7.6,
            family=FONT, zorder=6)


def load_pod(x, y, label):
    """Load box + ground, hanging directly below (x, y)."""
    box_y = y - 0.85
    box = FancyBboxPatch((x - 0.46, box_y - 0.20), 0.92, 0.40,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          linewidth=1.4, edgecolor=RED, facecolor=RED_FILL, zorder=5)
    ax.add_patch(box)
    ax.text(x, box_y, label, ha="center", va="center", color=RED, fontsize=8.4,
            fontweight="bold", family=FONT, zorder=6)
    vline(x, y - 0.22, box_y + 0.20)
    vline(x, box_y - 0.20, box_y - 0.42)
    ground(x, box_y - 0.42)


def side_load(x, y, label, side=1):
    """Load box + ground, hanging to one side (side=+1 right, -1 left) of (x, y)."""
    lx = x + side * 1.05
    hline(x, lx - side * 0.46, y)
    box = FancyBboxPatch((lx - 0.46, y - 0.20), 0.92, 0.40,
                          boxstyle="round,pad=0.02,rounding_size=0.05",
                          linewidth=1.4, edgecolor=RED, facecolor=RED_FILL, zorder=5)
    ax.add_patch(box)
    ax.text(lx, y, label, ha="center", va="center", color=RED, fontsize=8.4,
            fontweight="bold", family=FONT, zorder=6)
    vline(lx, y - 0.20, y - 0.42)
    ground(lx, y - 0.42)


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------
ax.text(7.5, 8.72, "IEEE 13-Node Test Feeder (Modified Reconstruction)", ha="center", va="top",
        color=FG_TITLE, fontsize=18, fontweight="bold", family=FONT)
ax.text(7.5, 8.32, "Radial distribution with two transformers (T1: Yg–Δ, T2: Δ–Δ) · Single-Line Diagram",
        ha="center", va="top", color=FG_SUB, fontsize=11, family=FONT)

# ---------------------------------------------------------------------------
# Zone backgrounds
# ---------------------------------------------------------------------------
x_t1 = 2.5
ax.add_patch(FancyBboxPatch((0.25, 5.55), x_t1 - 0.25, 2.15, boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=1.4, linestyle="--", edgecolor=ZONE_115_EDGE,
                             facecolor=ZONE_115_FILL, alpha=0.55, zorder=0))
ax.add_patch(FancyBboxPatch((x_t1, 0.95), 14.75 - x_t1, 6.75, boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=1.4, linestyle="--", edgecolor=ZONE_416_EDGE,
                             facecolor=ZONE_416_FILL, alpha=0.40, zorder=0))
ax.add_patch(FancyBboxPatch((3.55, 0.90), 1.65, 3.40, boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=1.5, linestyle="--", edgecolor=ZONE_048_EDGE,
                             facecolor=ZONE_048_FILL, alpha=0.85, zorder=1))

ax.text(1.4, 7.62, "115 kV · Source", ha="center", va="top", color=ZONE_115_EDGE, fontsize=8.6,
        family=FONT, fontweight="bold", zorder=1)
ax.text(14.5, 7.62, "4.16 kV · Primary distribution", ha="right", va="top", color=ZONE_416_EDGE,
        fontsize=9.5, family=FONT, fontweight="bold", zorder=1)
ax.text(5.1, 4.22, "0.48 kV · Secondary (T2)", ha="right", va="top", color=ZONE_048_EDGE,
        fontsize=7.6, family=FONT, fontweight="bold", zorder=2)

# ---------------------------------------------------------------------------
# Backbone: SourceBus -> T1 -> 650 -> 632 -> 671
# ---------------------------------------------------------------------------
Y_BB = 6.7
x_src, x_650, x_632, x_671 = 1.15, 3.9, 5.8, 10.6

ax.add_patch(Circle((x_src, Y_BB), 0.32, facecolor="none", edgecolor=GREEN, linewidth=2.2, zorder=5))
ax.plot([x_src - 0.17, x_src - 0.06, x_src + 0.06, x_src + 0.17],
        [Y_BB - 0.02, Y_BB + 0.12, Y_BB - 0.12, Y_BB + 0.02], color=GREEN, linewidth=1.7, zorder=6)
ax.text(x_src, Y_BB + 0.56, "Vs\n3ϕ slack", ha="center", va="bottom", color=GREEN, fontsize=8.6,
        family=FONT, fontweight="bold", zorder=6)
ax.text(x_src, Y_BB - 0.50, "SourceBus", ha="center", va="top", color=BLUE, fontsize=8.6,
        fontweight="bold", family=FONT, zorder=6)
vline(x_src, Y_BB - 0.32, Y_BB - 0.68)
ground(x_src, Y_BB - 0.68)
ax.text(x_src + 0.6, Y_BB - 0.05, "115 kV", ha="left", va="center", color=FG_SUB, fontsize=7.6,
        family=FONT, zorder=6)

hline(x_src + 0.32, x_t1 - 0.49, Y_BB)
transformer(x_t1, Y_BB, "Yg – Δ", "115 / 4.16 kV")

hline(x_t1 + 0.49, x_650, Y_BB)
bus_tick(x_650, Y_BB, "650")

hline(x_650, x_632, Y_BB)
h_label(x_650, x_632, Y_BB + 0.35, "Z650-632")
bus_tick(x_632, Y_BB, "632")

hline(x_632, x_671, Y_BB)
h_label(x_632, x_671, Y_BB + 0.35, "Z632-671 (670 pass-through)", fontsize=7.6)
bus_tick(x_671, Y_BB, "671")

# ---------------------------------------------------------------------------
# Lateral A: 632 -> 633 -> T2 -> 634  (0.48 kV secondary)
# ---------------------------------------------------------------------------
x_A = 4.375
y_633 = 5.15
vline(x_632, Y_BB - 0.22, y_633)
hline(x_632, x_A, y_633)
v_label(x_632, Y_BB - 0.22, y_633, "Z632-633", side=-1)
bus_tick(x_A, y_633, "633")

y_t2 = 4.0
vline(x_A, y_633 - 0.22, y_t2 + 0.32)
transformer(x_A, y_t2, "Δ – Δ", "4.16 / 0.48 kV")

y_634 = 2.55
vline(x_A, y_t2 - 0.50, y_634 + 0.22)
bus_tick(x_A, y_634, "634", label_pos="left")
load_pod(x_A, y_634, "Z_634")

# ---------------------------------------------------------------------------
# Lateral B: 632 -> 645 -> 646
# ---------------------------------------------------------------------------
x_B = 7.3
y_645 = 5.15
vline(x_632, Y_BB - 0.22, y_645)
hline(x_632, x_B, y_645)
v_label(x_632, Y_BB - 0.22, y_645, "Z632-645", side=1)
bus_tick(x_B, y_645, "645")
side_load(x_B, y_645, "Z_645", side=1)

y_646 = 3.55
vline(x_B, y_645 - 0.22, y_646 + 0.22)
v_label(x_B, y_646 + 0.22, y_645 - 0.22, "Z645-646", side=1)
bus_tick(x_B, y_646, "646")
side_load(x_B, y_646, "Z_646", side=1)

# ---------------------------------------------------------------------------
# Lateral C: 671 -> 675 (single load)
# ---------------------------------------------------------------------------
x_C = 9.2
y_675 = 5.15
vline(x_671, Y_BB - 0.22, y_675)
hline(x_671, x_C, y_675)
v_label(x_671, Y_BB - 0.22, y_675, "Z671-675", side=-1)
bus_tick(x_C, y_675, "675", label_pos="right")
load_pod(x_C, y_675, "Z_675")

# ---------------------------------------------------------------------------
# Lateral D: 671 -> 684 -> {611, 652}
# ---------------------------------------------------------------------------
x_D = 12.4
y_684 = 5.15
vline(x_671, Y_BB - 0.22, y_684)
hline(x_671, x_D, y_684)
v_label(x_671, Y_BB - 0.22, y_684, "Z671-684", side=1)
bus_tick(x_D, y_684, "684")

x_611, x_652 = 11.35, 13.45
y_leaf = 3.55
hline(x_611, x_652, y_leaf)
vline(x_D, y_684 - 0.22, y_leaf)
v_label(x_D, y_leaf, y_684 - 0.22, "", side=1)
h_label(x_D, x_652, y_leaf + 0.35, "Z684-652", boxed=False)
h_label(x_611, x_D, y_leaf + 0.35, "Z684-611", boxed=False)
bus_tick(x_611, y_leaf, "611", label_pos="left")
load_pod(x_611, y_leaf, "Z_611")
bus_tick(x_652, y_leaf, "652", label_pos="right")
load_pod(x_652, y_leaf, "Z_652")

# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------
caption = (
    "SourceBus → T1 (Yg–Δ) → 650 → 632, feeding the 671 backbone (→675, →684→{611,652}) plus two laterals\n"
    "off 632: 632→645→646, and 632→633→T2 (Δ–Δ)→634. T1/T2's floating Δ windings are this project's documented\n"
    "source of Jacobian rank-deficiency (docs/project_guide.tex §6). Bus 670 is folded into segment 632-671 and\n"
    "dead-end bus 680 is omitted -- schematic simplifications, not topology changes."
)
ax.text(7.5, 0.12, caption, ha="center", va="bottom", color=FG_SUB, fontsize=8.6, family=FONT)

fig.tight_layout(pad=0.6)

out_jpg = os.path.join(FIGDIR, "ieee13.jpg")
out_pdf = os.path.join(FIGDIR, "ieee13.pdf")
fig.savefig(out_jpg, facecolor=BG)
fig.savefig(out_pdf, facecolor=BG)
plt.close(fig)
print(f"wrote {out_jpg}")
print(f"wrote {out_pdf}")
