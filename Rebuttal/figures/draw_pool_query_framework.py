"""Draw the two fixed-pool selection routes at manuscript width."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


def draw(output: Path) -> None:
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "font.size": 8.5,
        "text.color": "#1c2025",
    })
    fig = plt.figure(figsize=(5.5, 2.31))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 5.5), ylim=(0, 2.31), aspect="equal")
    ax.set_axis_off()

    ink, grey, blue = "#1c2025", "#565e66", "#244d62"
    light_grey, light_blue = "#f3f4f4", "#edf3f5"

    def line(x0, y0, x1, y1, color=grey, style="-"):
        ax.plot([x0, x1], [y0, y1], color=color, lw=0.65, ls=style)

    def arrow(x0, y0, x1, y1, color=grey, style="-"):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5.8,
            color=color, linewidth=0.7, linestyle=style, shrinkA=0, shrinkB=0,
        ))

    def box(x, y, width, height, title, detail, edge=grey, fill="white"):
        ax.add_patch(Rectangle((x, y), width, height, facecolor=fill,
                               edgecolor=edge, linewidth=0.65))
        title_y = y + (height * 0.67 if detail else height / 2)
        ax.text(x + width / 2, title_y, title, ha="center", va="center",
                fontweight="bold", fontsize=8.5)
        if detail:
            ax.text(x + width / 2, y + height * 0.31, detail, ha="center",
                    va="center", fontsize=8.1)

    # Shared fixed inventory, with one square per pool.
    for pool in range(15):
        col, row = pool % 5, pool // 5
        ax.add_patch(Rectangle((0.206 + col * 0.123, 1.51 + row * 0.117),
                               0.087, 0.077, facecolor=light_grey,
                               edgecolor=grey, linewidth=0.5))
    ax.text(0.50, 1.343, "15 source pools", ha="center", va="center", weight="bold")
    ax.text(0.50, 1.175, "455 candidate triples", ha="center", va="center", fontsize=8.1)
    line(0.94, 1.44, 1.075, 1.44)
    line(1.075, 1.00, 1.075, 1.86)
    arrow(1.075, 1.86, 1.21, 1.86)
    arrow(1.075, 1.00, 1.21, 1.00, color=blue)

    # The two policies branch from the inventory and meet only at final selection.
    ax.text(1.21, 2.177, "Static feature screening", weight="bold",
            va="center", color=grey, fontsize=9.0)
    box(1.21, 1.65, 1.46, 0.42, "Rank triples", "Target A-opt or MMD", fill=light_grey)
    arrow(2.67, 1.86, 2.88, 1.86)
    box(2.88, 1.65, 1.42, 0.42, "Fit and score shortlist", "", fill=light_grey)
    ax.text(1.22, 1.547, "Source and unlabeled target features", fontsize=8.0, va="center")

    ax.text(1.21, 1.314, "Validation-guided pool querying", weight="bold",
            va="center", color=blue, fontsize=9.0)
    box(1.21, 0.79, 0.86, 0.42, "Fit and score 15", "Pivoted basis", blue, light_blue)
    arrow(2.07, 1.00, 2.23, 1.00, color=blue)
    box(2.23, 0.79, 1.01, 0.42, "Estimate effects", "From validation loss", blue, light_blue)
    arrow(3.24, 1.00, 3.40, 1.00, color=blue)
    box(3.40, 0.79, 0.90, 0.42, "Choose and fit", "Loss and leverage", blue, light_blue)

    line(4.30, 1.86, 4.45, 1.86)
    line(4.30, 1.00, 4.45, 1.00, color=blue)
    line(4.45, 1.00, 4.45, 1.43, color=blue)
    line(4.45, 1.43, 4.45, 1.86)
    arrow(4.45, 1.43, 4.60, 1.43, color=ink)
    box(4.60, 1.22, 0.86, 0.42, "Keep best fitted", "by validation loss", ink)
    arrow(5.03, 1.22, 5.03, 0.93, style="--")
    ax.text(5.03, 0.80, "Held-out test", ha="center", va="center", fontsize=8.1)

    # The example makes partial overlap and the approximate swap interpretation visible.
    line(3.92, 0.79, 3.92, 0.705, color=blue)
    line(3.92, 0.705, 2.97, 0.705, color=blue)
    arrow(2.97, 0.705, 2.97, 0.79, color=blue)
    ax.text(3.92, 0.595, "Refit if sequential", ha="center", va="center",
            fontsize=7.7, color=blue)
    line(2.425, 0.79, 2.425, 0.674, color=blue)
    ax.text(2.42, 0.557, r"$A+B+C$  vs.  $A+B+D$", ha="center",
            va="center", color=blue, fontsize=9.4)
    ax.text(2.42, 0.392, r"Loss difference estimates the effect of exchanging $C$ and $D$",
            ha="center", va="center", fontsize=8.1)
    line(0.06, 0.249, 5.44, 0.249, color="#adb3b7")
    ax.text(2.75, 0.125, "Fit on labeled source pools; score on labeled target validation data.",
            ha="center", va="center", fontsize=8.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Output figure path (PDF recommended)")
    draw(parser.parse_args().output)
