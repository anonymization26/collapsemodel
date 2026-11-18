"""Plot test Brier gaps for the paper's fresh logistic pool-query experiments."""

import argparse
import csv
import math
from pathlib import Path
from statistics import fmean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


INK = "#222222"
BLUE = "#254f69"
GRAY = "#6f7073"
GOLD = "#996746"
DOMAINNET_METHODS = (
    ("Sequential 25", "sequential25", BLUE),
    ("Two-batch 25", "batch25", BLUE),
    ("D-design 25", "d", GRAY),
    ("MMD 50", "mmd50", GOLD),
)
OPTDIGITS_METHODS = (
    ("Sequential 25", "sequential25", BLUE),
    ("XY-design 25", "xy", GRAY),
    ("MMD 25", "mmd25", GOLD),
)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def gap_series(rows, dataset, method, expected_units, partition=None):
    prefix = dataset if partition is None else f"{dataset}_{partition}"
    matches = [
        row for row in rows
        if row["unit"].startswith(prefix + "_")
        and row["method"] == method
        and row.get("finish", "recommend") == "recommend"
        and row.get("comparison", "standard") == "standard"
        and row["budget"] == ("50" if method == "mmd50" else "25")
    ]
    by_unit = {row["unit"]: row for row in matches}
    if len(matches) != len(by_unit) or by_unit.keys() != expected_units:
        raise ValueError(f"Mismatched units for {prefix}, {method}")
    values = []
    for name in sorted(by_unit):
        row = by_unit[name]
        gap = float(row["gap_full"])
        if not math.isclose(gap, float(row["brier"]) - float(row["full_brier"]), abs_tol=1e-12):
            raise ValueError(f"Invalid full-validation gap for {name}, {method}")
        values.append(1000 * gap)
    return values


def format_axis(axis, limits, ticks):
    axis.set_xlim(limits)
    axis.set_xticks(ticks)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_linewidth(0.55)
    axis.spines["bottom"].set_color(GRAY)
    axis.tick_params(axis="x", length=2.5, width=0.55, pad=2)
    axis.tick_params(axis="y", length=0, pad=4)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#e4e4e4", linewidth=0.45)
    axis.axvline(0, color=INK, linewidth=0.65, zorder=1)
    axis.set_xlabel(r"Test Brier gap ($\times 10^{-3}$)", labelpad=4)


def check_visible(axis, values, method):
    lower, upper = axis.get_xlim()
    if min(values) <= lower or max(values) >= upper:
        raise ValueError(f"Data for {method} exceed axis limits {lower:g}, {upper:g}")


def plot_domainnet(axis, fresh, linear, units):
    format_axis(axis, (-4.1, 15.3), [-4, 0, 5, 10, 15])
    axis.set_ylim(-0.52, 4.02)
    axis.set_yticks([3, 2, 1, 0], [label for label, _, _ in DOMAINNET_METHODS])
    axis.axvline(5, color=GRAY, linewidth=0.68, linestyle=(0, (3, 2)), zorder=1)
    axis.text(5.24, 3.82, "+0.005 threshold", ha="left", va="center", color=GRAY, fontsize=7)
    for y, (_, method, color) in zip([3, 2, 1, 0], DOMAINNET_METHODS):
        values = gap_series(linear if method == "d" else fresh, "domainnet", method, units)
        check_visible(axis, values, method)
        jitter = [y + ((7 * i % 19) - 9) * 0.024 for i in range(len(values))]
        axis.scatter(values, jitter, s=5, color=color, alpha=0.34, linewidths=0, zorder=2)
        axis.scatter(fmean(values), y, s=24, marker="D", color=color,
                     edgecolor="white", linewidth=0.58, zorder=3)


def plot_optdigits(axis, fresh, linear, partitions):
    format_axis(axis, (-5.5, 87.0), [0, 20, 40, 60, 80])
    axis.set_ylim(-0.62, 7.55)
    axis.set_yticks([6, 5, 4, 2, 1, 0], [label for label, _, _ in OPTDIGITS_METHODS] * 2)
    axis.axhline(3.16, color="#d9d9d9", linewidth=0.5)
    for (partition, units), y_rows, label in zip(
        partitions.items(), ([6, 5, 4], [2, 1, 0]), ("Hash partition", "PC1 partition")
    ):
        axis.text(0.985, y_rows[0] + 0.68, label, transform=axis.get_yaxis_transform(),
                  ha="right", va="center", fontsize=7, color=GRAY)
        for y, (_, method, color) in zip(y_rows, OPTDIGITS_METHODS):
            values = gap_series(linear if method == "xy" else fresh, "optdigits",
                                method, units, partition)
            check_visible(axis, values, method)
            mean = fmean(values)
            axis.hlines(y, min(values), max(values), color=color, linewidth=0.75, alpha=0.7)
            axis.scatter(values, [y] * len(values), marker="|", s=15, color=color,
                         linewidth=0.58, alpha=0.8, zorder=2)
            axis.scatter(mean, y, s=24, marker="D", color=color,
                         edgecolor="white", linewidth=0.58, zorder=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-csv", type=Path, required=True)
    parser.add_argument("--linear-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fresh = read_csv(args.fresh_csv)
    linear = read_csv(args.linear_csv)
    full = {row["unit"] for row in fresh if row["method"] == "full455"}
    domainnet_units = {name for name in full if name.startswith("domainnet_")}
    partitions = {
        part: {name for name in full if name.startswith(f"optdigits_{part}_")}
        for part in ("hash", "pc1")
    }
    if len(domainnet_units) != 48 or any(len(units) != 4 for units in partitions.values()):
        raise ValueError("Expected 48 DomainNet configurations and four seeds per Optdigits partition")

    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(5.5, 2.24))
    domainnet = fig.add_axes([0.14, 0.20, 0.375, 0.68])
    optdigits = fig.add_axes([0.655, 0.20, 0.328, 0.68])
    fig.text(0.01, 0.935, "(a)  DomainNet", fontsize=9, fontweight="bold")
    fig.text(0.53, 0.935, "(b)  Optdigits", fontsize=9, fontweight="bold")
    plot_domainnet(domainnet, fresh, linear, domainnet_units)
    plot_optdigits(optdigits, fresh, linear, partitions)
    fig.savefig(args.output, dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    main()
