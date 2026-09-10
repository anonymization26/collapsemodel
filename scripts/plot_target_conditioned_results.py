#!/usr/bin/env python3
"""Generate E1 and E5/H5 figures directly from raw experiment tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "collapsemodel-matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "collapsemodel-cache")
)
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


CONFIGURATION_FIELDS = (
    "seed",
    "dimension",
    "candidate_count",
    "budget",
    "target_samples",
    "target_rank",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return float(np.mean(items)) if items else math.nan


def config_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in CONFIGURATION_FIELDS)


def target_rank_fraction(row: dict[str, str]) -> float:
    return int(row["target_rank"]) / int(row["dimension"])


def plot_h2_heatmap(
    shared_rows: list[dict[str, str]],
    summary_report: dict[str, object],
    output: Path,
    gate_name: str = "h2_pilot",
) -> None:
    experiment_summary = summary_report["summary"]
    gate = experiment_summary["gates"][gate_name]
    baseline_field = (
        "strongest_target_blind_baseline"
        if gate_name == "h2_pilot"
        else "strongest_same_information_baseline"
    )
    baseline = gate[baseline_field]
    target = {
        config_key(row): row
        for row in shared_rows
        if row["method"] == "target_a_estimated"
    }
    baseline_rows = {
        config_key(row): row
        for row in shared_rows
        if row["method"] == baseline
    }
    grouped: dict[tuple[float, int], list[float]] = defaultdict(list)
    for key, target_row in target.items():
        baseline_row = baseline_rows[key]
        baseline_risk = float(baseline_row["target_risk"])
        improvement = (
            baseline_risk - float(target_row["target_risk"])
        ) / max(abs(baseline_risk), 1e-15)
        grouped[(
            target_rank_fraction(target_row),
            int(target_row["target_samples"]),
        )].append(improvement)
    rank_fractions = sorted({key[0] for key in grouped})
    target_samples = sorted({key[1] for key in grouped})
    matrix = np.array([
        [mean(grouped[(fraction, samples)]) for samples in target_samples]
        for fraction in rank_fractions
    ])

    figure, axis = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    limit = max(float(np.nanmax(np.abs(matrix))), 0.01)
    image = axis.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(target_samples)), labels=target_samples)
    axis.set_yticks(
        range(len(rank_fractions)),
        labels=[f"{fraction:.3g}" for fraction in rank_fractions],
    )
    axis.set_xlabel("Unlabeled target samples")
    axis.set_ylabel("Target rank / dimension")
    axis.set_title(f"Target A-opt risk improvement vs {baseline}")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            color = "white" if abs(value) > 0.55 * limit else "black"
            axis.text(
                column_index,
                row_index,
                f"{100 * value:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Mean relative risk improvement")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_target_estimation_regret(
    shared_rows: list[dict[str, str]],
    output: Path,
) -> None:
    rows = [row for row in shared_rows if row["method"] == "target_a_estimated"]
    grouped: dict[tuple[float, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(
            target_rank_fraction(row),
            int(row["target_samples"]),
        )].append(float(row["normalized_regret"]))
    rank_fractions = sorted({key[0] for key in grouped})
    sample_counts = sorted({key[1] for key in grouped})
    figure, axis = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    for fraction in rank_fractions:
        values = [mean(grouped[(fraction, count)]) for count in sample_counts]
        axis.plot(sample_counts, values, marker="o", label=f"rT/d={fraction:.3g}")
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Unlabeled target samples")
    axis.set_ylabel("Mean normalized regret")
    axis.set_title("Target covariance estimation sensitivity")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_conditional_shift(
    shift_rows: list[dict[str, str]],
    output: Path,
) -> None:
    selected_methods = {
        "target_a_estimated",
        "target_energy",
        "second_moment_mmd",
        "effective_rank",
        "isotropic_a",
        "bayesian_d",
    }
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in shift_rows:
        if row["method"] in selected_methods:
            grouped[(row["method"], float(row["shift"]))].append(
                float(row["normalized_regret"])
            )
    shifts = sorted({key[1] for key in grouped})
    figure, axis = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    for method in sorted(selected_methods):
        values = [mean(grouped[(method, shift)]) for shift in shifts]
        axis.plot(shifts, values, marker="o", label=method)
    axis.set_xlabel("Conditional-shift magnitude")
    axis.set_ylabel("Mean normalized regret")
    axis.set_title("Failure boundary under conditional shift")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_h5_tradeoff(summary_report: dict[str, object], output: Path) -> None:
    strategies = summary_report["summary"]["strategies"]
    points = [
        {"strategy_id": name, **values}
        for name, values in strategies.items()
        if values["mode"] != "full_gram" and values["covers_full_grid"]
    ]
    colors = {
        "fixed_rank": "#3B6FB6",
        "adaptive_capped": "#D17A22",
        "adaptive_full_fallback": "#2E8B57",
    }
    markers = {
        "fixed_rank": "o",
        "adaptive_capped": "x",
        "adaptive_full_fallback": "^",
    }
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    for mode, color in colors.items():
        subset = [point for point in points if point["mode"] == mode]
        axes[0].scatter(
            [point["mean_byte_ratio_vs_packed"] for point in subset],
            [point["pair_certificate_rate"] for point in subset],
            color=color,
            label=mode,
            s=42,
            marker=markers[mode],
        )
        axes[1].scatter(
            [point["mean_byte_ratio_vs_packed"] for point in subset],
            [point["selection_match_rate"] for point in subset],
            color=color,
            label=mode,
            s=42,
            marker=markers[mode],
        )
        grouped_labels: dict[tuple[float, float], list[object]] = defaultdict(list)
        for point in subset:
            grouped_labels[(
                float(point["mean_byte_ratio_vs_packed"]),
                float(point["pair_certificate_rate"]),
            )].append(point["rank_cap"])
        for (x_value, y_value), ranks in grouped_labels.items():
            label = "L=" + ",".join(str(rank) for rank in ranks)
            axes[0].annotate(
                label,
                (x_value, y_value),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
        grouped_labels = defaultdict(list)
        for point in subset:
            grouped_labels[(
                float(point["mean_byte_ratio_vs_packed"]),
                float(point["selection_match_rate"]),
            )].append(point["rank_cap"])
        for (x_value, y_value), ranks in grouped_labels.items():
            label = "L=" + ",".join(str(rank) for rank in ranks)
            axes[1].annotate(
                label,
                (x_value, y_value),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
    axes[0].axvline(0.25, color="black", linestyle="--", linewidth=1)
    axes[0].axhline(0.50, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Direct comparison certificates")
    axes[0].set_ylabel("Pair certificate rate")
    axes[1].axvline(0.50, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Final selection fidelity")
    axes[1].set_ylabel("Selection match rate")
    maximum_ratio = max(
        float(point["mean_byte_ratio_vs_packed"]) for point in points
    )
    for axis in axes:
        axis.set_xlabel("Bytes / packed full-Gram bytes")
        axis.grid(alpha=0.25)
        axis.set_ylim(-0.03, 1.05)
        axis.set_xlim(left=0.0, right=max(0.55, 1.12 * maximum_ratio))
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-dir", type=Path)
    parser.add_argument("--h5-dir", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "target_conditioned" / "figures",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.e1_dir is None and args.h5_dir is None:
        raise ValueError("provide --e1-dir, --h5-dir, or both")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    sources = []
    if args.e1_dir is not None:
        shared_path = args.e1_dir / "shared_model_results.csv"
        shift_path = args.e1_dir / "conditional_shift_results.csv"
        summary_path = args.e1_dir / "summary.json"
        shared_rows = read_csv(shared_path)
        shift_rows = read_csv(shift_path)
        summary = read_json(summary_path)
        figure_specs = [
            (
                "e1_h2_target_blind_improvement_heatmap.png",
                lambda path: plot_h2_heatmap(shared_rows, summary, path),
            ),
            (
                "e1_h2_same_information_improvement_heatmap.png",
                lambda path: plot_h2_heatmap(
                    shared_rows,
                    summary,
                    path,
                    gate_name="h2_same_information",
                ),
            ),
            (
                "e1_target_estimation_regret.png",
                lambda path: plot_target_estimation_regret(shared_rows, path),
            ),
            (
                "e1_conditional_shift.png",
                lambda path: plot_conditional_shift(shift_rows, path),
            ),
        ]
        for name, plotter in figure_specs:
            output = args.out_dir / name
            plotter(output)
            outputs.append(output)
        sources.extend([shared_path, shift_path, summary_path])
    if args.h5_dir is not None:
        summary_path = args.h5_dir / "summary.json"
        output = args.out_dir / "e5_h5_tradeoff.png"
        plot_h5_tradeoff(read_json(summary_path), output)
        outputs.append(output)
        sources.append(summary_path)

    manifest = {
        "sources": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in sources
        ],
        "outputs": [
            {"path": path.name, "sha256": sha256_file(path)}
            for path in outputs
        ],
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "completed",
        "figures": [str(path) for path in outputs],
        "manifest": str(manifest_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
