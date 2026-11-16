#!/usr/bin/env python3
"""E5/H5 synthetic rank, certificate, and communication sweep."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_target_conditioned_e1 as e1  # noqa: E402
from metrics.target_conditioned import (  # noqa: E402
    AdaptiveSketchResult,
    PSDEigendecomposition,
    adaptive_sketch_target_a,
    aggregate_gram,
    candidate_sketch_intervals,
    decompose_psd,
    greedy_target_a,
    sketch_from_decomposition,
    target_a_objective,
)


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return float(np.mean(items)) if items else math.nan


def packed_gram_bytes(candidate_count: int, dimension: int) -> int:
    entries = dimension * (dimension + 1) // 2
    return candidate_count * entries * np.dtype(np.float64).itemsize


def risk_for_selection(
    problem: e1.SyntheticProblem,
    selected: Iterable[str],
    target_moment: np.ndarray,
) -> float:
    return target_a_objective(
        target_moment,
        problem.prior_precision,
        aggregate_gram(problem.blocks, selected),
        problem.noise_variance,
    )


def adaptive_interval_diagnostics(
    problem: e1.SyntheticProblem,
    decompositions: Mapping[str, PSDEigendecomposition],
    result: AdaptiveSketchResult,
) -> dict[str, float | int]:
    names = sorted(problem.blocks)
    selected: list[str] = []
    covered_intervals = 0
    total_intervals = 0
    certified_comparisons = 0
    total_comparisons = 0
    widths = []
    for step in result.steps:
        ranks = dict(step.requested_ranks)
        sketches = {
            name: sketch_from_decomposition(decompositions[name], ranks[name])
            for name in names
        }
        candidates = [name for name in names if name not in selected]
        intervals = candidate_sketch_intervals(
            selected,
            candidates,
            sketches,
            problem.estimated_target_moment,
            problem.prior_precision,
            problem.noise_variance,
        )
        chosen_interval = intervals[step.candidate]
        widths.append(chosen_interval.width)
        certified_comparisons += sum(
            chosen_interval.upper < intervals[name].lower
            for name in candidates
            if name != step.candidate
        )
        total_comparisons += max(len(candidates) - 1, 0)
        prefix_gram = aggregate_gram(problem.blocks, selected)
        for name in candidates:
            exact = target_a_objective(
                problem.estimated_target_moment,
                problem.prior_precision,
                prefix_gram + problem.blocks[name],
                problem.noise_variance,
            )
            interval = intervals[name]
            tolerance = 1e-10 * max(abs(exact), 1.0)
            covered_intervals += int(
                interval.lower - tolerance <= exact <= interval.upper + tolerance
            )
            total_intervals += 1
        selected.append(step.candidate)
    return {
        "certified_steps": sum(step.certified for step in result.steps),
        "certificate_rate": result.certificate_rate,
        "certified_comparisons": certified_comparisons,
        "total_comparisons": total_comparisons,
        "pair_certificate_rate": (
            certified_comparisons / total_comparisons
            if total_comparisons
            else 1.0
        ),
        "covered_intervals": covered_intervals,
        "total_intervals": total_intervals,
        "interval_coverage_rate": covered_intervals / total_intervals,
        "mean_selected_interval_width": mean(widths),
        "uncertified_fallback_steps": sum(
            step.used_uncertified_fallback for step in result.steps
        ),
        "mean_refinement_rounds": mean(
            step.refinement_rounds for step in result.steps
        ),
    }


def strategy_record(
    problem: e1.SyntheticProblem,
    config: dict[str, int],
    strategy_id: str,
    mode: str,
    selected: Iterable[str],
    full_selected: tuple[str, ...],
    full_proxy_risk: float,
    full_true_risk: float,
    transmitted_bytes: int,
    dense_full_bytes: int,
    packed_full_bytes: int,
    elapsed_seconds: float,
    decomposition_seconds: float,
    rank_cap: int | None = None,
    rank_schedule: Iterable[int] = (),
    fallback_to_full: bool = False,
    diagnostics: Mapping[str, float | int] | None = None,
) -> dict[str, object]:
    chosen = tuple(sorted(selected))
    proxy_risk = risk_for_selection(
        problem, chosen, problem.estimated_target_moment
    )
    true_risk = risk_for_selection(problem, chosen, problem.target_moment)
    record: dict[str, object] = {
        **config,
        "strategy_id": strategy_id,
        "mode": mode,
        "rank_cap": rank_cap,
        "rank_cap_fraction": (
            rank_cap / config["dimension"] if rank_cap is not None else None
        ),
        "rank_schedule": "|".join(str(rank) for rank in rank_schedule),
        "fallback_to_full": int(fallback_to_full),
        "selected": "|".join(chosen),
        "full_selected": "|".join(sorted(full_selected)),
        "selection_match": int(chosen == tuple(sorted(full_selected))),
        "proxy_risk": proxy_risk,
        "full_proxy_risk": full_proxy_risk,
        "relative_proxy_risk_vs_full_greedy": (
            (proxy_risk - full_proxy_risk) / max(abs(full_proxy_risk), 1e-15)
        ),
        "true_target_risk": true_risk,
        "full_true_target_risk": full_true_risk,
        "relative_true_risk_vs_full_greedy": (
            (true_risk - full_true_risk) / max(abs(full_true_risk), 1e-15)
        ),
        "transmitted_bytes": transmitted_bytes,
        "dense_full_gram_bytes": dense_full_bytes,
        "packed_full_gram_bytes": packed_full_bytes,
        "byte_ratio_vs_dense": transmitted_bytes / dense_full_bytes,
        "byte_ratio_vs_packed": transmitted_bytes / packed_full_bytes,
        "selection_seconds": elapsed_seconds,
        "decomposition_seconds": decomposition_seconds,
    }
    if diagnostics:
        record.update(diagnostics)
    return record


def summarize(
    rows: list[dict[str, object]],
    configuration_count: int,
) -> dict[str, object]:
    summaries = {}
    for strategy_id in sorted({str(row["strategy_id"]) for row in rows}):
        strategy_rows = [row for row in rows if row["strategy_id"] == strategy_id]
        first = strategy_rows[0]
        interval_rows = [
            row
            for row in strategy_rows
            if row.get("total_intervals") not in (None, "")
        ]
        total_intervals = sum(int(row["total_intervals"]) for row in interval_rows)
        total_comparisons = sum(
            int(row["total_comparisons"]) for row in interval_rows
        )
        summaries[strategy_id] = {
            "mode": first["mode"],
            "rank_cap": (
                int(first["rank_cap"])
                if first["rank_cap"] not in (None, "")
                else None
            ),
            "n_configurations": len(strategy_rows),
            "covers_full_grid": len(strategy_rows) == configuration_count,
            "selection_match_rate": mean(
                float(row["selection_match"]) for row in strategy_rows
            ),
            "mean_relative_proxy_risk_vs_full_greedy": mean(
                float(row["relative_proxy_risk_vs_full_greedy"])
                for row in strategy_rows
            ),
            "mean_relative_true_risk_vs_full_greedy": mean(
                float(row["relative_true_risk_vs_full_greedy"])
                for row in strategy_rows
            ),
            "mean_byte_ratio_vs_dense": mean(
                float(row["byte_ratio_vs_dense"]) for row in strategy_rows
            ),
            "mean_byte_ratio_vs_packed": mean(
                float(row["byte_ratio_vs_packed"]) for row in strategy_rows
            ),
            "mean_selection_seconds": mean(
                float(row["selection_seconds"]) for row in strategy_rows
            ),
        }
        if interval_rows:
            summaries[strategy_id].update({
                "interval_coverage_rate": (
                    sum(int(row["covered_intervals"]) for row in interval_rows)
                    / total_intervals
                ),
                "pair_certificate_rate": (
                    sum(
                        int(row["certified_comparisons"])
                        for row in interval_rows
                    )
                    / total_comparisons
                    if total_comparisons
                    else 1.0
                ),
                "fully_certified_selection_rate": mean(
                    float(float(row["certificate_rate"]) == 1.0)
                    for row in interval_rows
                ),
                "mean_certificate_rate": mean(
                    float(row["certificate_rate"]) for row in interval_rows
                ),
                "mean_interval_width": mean(
                    float(row["mean_selected_interval_width"])
                    for row in interval_rows
                ),
            })

    sketch_summaries = [
        {"strategy_id": strategy_id, **values}
        for strategy_id, values in summaries.items()
        if values["mode"] != "full_gram"
    ]
    coverage_ok = all(
        float(values["interval_coverage_rate"]) == 1.0
        for values in sketch_summaries
    )
    direct_candidates = [
        values
        for values in sketch_summaries
        if values["covers_full_grid"]
        and values["mode"] in {"fixed_rank", "adaptive_capped"}
        and float(values["mean_byte_ratio_vs_packed"]) <= 0.25
        and float(values["pair_certificate_rate"]) >= 0.50
        and float(values["selection_match_rate"]) >= 0.90
    ]
    fallback_candidates = [
        values
        for values in sketch_summaries
        if values["covers_full_grid"]
        and values["mode"] == "adaptive_full_fallback"
        and float(values["mean_byte_ratio_vs_packed"]) <= 0.50
        and float(values["selection_match_rate"]) == 1.0
    ]
    best_direct = (
        min(
            direct_candidates,
            key=lambda item: (
                float(item["mean_byte_ratio_vs_packed"]),
                -float(item["pair_certificate_rate"]),
            ),
        )
        if direct_candidates
        else None
    )
    best_fallback = (
        min(
            fallback_candidates,
            key=lambda item: float(item["mean_byte_ratio_vs_packed"]),
        )
        if fallback_candidates
        else None
    )
    passed = coverage_ok and (best_direct is not None or best_fallback is not None)
    return {
        "strategies": summaries,
        "h5_gate": {
            "status": "passed" if passed else "failed",
            "byte_denominator": "packed symmetric float64 Gram",
            "all_deterministic_intervals_cover_full_gram_risk": coverage_ok,
            "direct_path": {
                "passed": best_direct is not None,
                "qualifying_strategy": best_direct,
                "requirements": {
                    "mean_byte_ratio_at_most": 0.25,
                    "pair_certificate_rate_at_least": 0.50,
                    "selection_match_rate_at_least": 0.90,
                },
            },
            "adaptive_fallback_path": {
                "passed": best_fallback is not None,
                "qualifying_strategy": best_fallback,
                "requirements": {
                    "mean_byte_ratio_at_most": 0.50,
                    "selection_match_rate": 1.0,
                },
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "target_conditioned" / "e5_sketch_certificate",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260910])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--candidate-counts", type=int, nargs="+", default=[8, 12])
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--target-samples", type=int, nargs="+", default=[32, 128])
    parser.add_argument(
        "--target-rank-fractions", type=float, nargs="+", default=[0.125, 0.25, 0.5]
    )
    parser.add_argument("--source-samples", type=int, default=64)
    parser.add_argument(
        "--sketch-ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument("--mini", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    integer_lists = [
        args.seeds,
        args.dimensions,
        args.candidate_counts,
        args.budgets,
        args.target_samples,
        args.sketch_ranks,
    ]
    if any(not values for values in integer_lists):
        raise ValueError("grid arguments must be nonempty")
    if any(value <= 0 for values in integer_lists[1:] for value in values):
        raise ValueError("dimensions, counts, budgets, samples, and ranks must be positive")
    if any(dimension < 4 for dimension in args.dimensions):
        raise ValueError("dimensions must be at least 4")
    if args.source_samples <= 0:
        raise ValueError("source samples must be positive")
    if any(
        not math.isfinite(fraction) or fraction <= 0.0 or fraction > 0.5
        for fraction in args.target_rank_fractions
    ):
        raise ValueError("target rank fractions must lie in (0, 0.5]")
    for count in args.candidate_counts:
        if any(budget > count for budget in args.budgets):
            raise ValueError("selection budget cannot exceed candidate count")
    for dimension in args.dimensions:
        target_ranks = [
            max(1, min(int(round(dimension * fraction)), dimension // 2))
            for fraction in args.target_rank_fractions
        ]
        if len(set(target_ranks)) != len(target_ranks):
            raise ValueError(
                "target rank fractions collapse to duplicate ranks at "
                f"dimension {dimension}: {target_ranks}"
            )


def main() -> int:
    args = parse_args()
    if args.mini:
        args.seeds = args.seeds[:1]
        args.dimensions = [min(args.dimensions)]
        args.candidate_counts = [min(args.candidate_counts)]
        args.budgets = [min(args.budgets)]
        args.target_samples = [min(args.target_samples)]
        args.target_rank_fractions = [0.25]
        args.sketch_ranks = [1, 2, min(4, min(args.dimensions))]
    validate_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows: list[dict[str, object]] = []
    total_configurations = (
        len(args.seeds)
        * len(args.dimensions)
        * len(args.candidate_counts)
        * len(args.budgets)
        * len(args.target_samples)
        * len(args.target_rank_fractions)
    )
    completed = 0
    for seed in args.seeds:
        for dimension in args.dimensions:
            valid_ranks = sorted({
                rank for rank in args.sketch_ranks if rank <= dimension
            })
            if not valid_ranks:
                raise ValueError(f"no sketch rank is valid for dimension {dimension}")
            for candidate_count in args.candidate_counts:
                for target_samples in args.target_samples:
                    for target_rank_fraction in args.target_rank_fractions:
                        target_rank = max(
                            1,
                            min(
                                int(round(dimension * target_rank_fraction)),
                                dimension // 2,
                            ),
                        )
                        problem = e1.make_synthetic_problem(
                            seed=e1.synthetic_problem_seed(
                                seed,
                                dimension,
                                candidate_count,
                            ),
                            dimension=dimension,
                            candidate_count=candidate_count,
                            source_samples=args.source_samples,
                            target_samples=target_samples,
                            target_rank=target_rank,
                        )
                        decomposition_started = time.monotonic()
                        decompositions = {
                            name: decompose_psd(gram)
                            for name, gram in problem.blocks.items()
                        }
                        decomposition_seconds = (
                            time.monotonic() - decomposition_started
                        )
                        dense_bytes = candidate_count * dimension**2 * 8
                        packed_bytes = packed_gram_bytes(
                            candidate_count, dimension
                        )
                        estimation_error = float(
                            np.linalg.norm(
                                problem.estimated_target_moment
                                - problem.target_moment,
                                ord="fro",
                            )
                            / max(
                                np.linalg.norm(problem.target_moment, ord="fro"),
                                np.finfo(float).tiny,
                            )
                        )
                        for budget in args.budgets:
                            config = {
                                "seed": seed,
                                "dimension": dimension,
                                "candidate_count": candidate_count,
                                "budget": budget,
                                "target_samples": target_samples,
                                "target_rank": target_rank,
                                "target_moment_relative_error": estimation_error,
                            }
                            full_started = time.monotonic()
                            full = greedy_target_a(
                                problem.blocks,
                                problem.estimated_target_moment,
                                problem.prior_precision,
                                budget,
                                problem.noise_variance,
                            )
                            full_seconds = time.monotonic() - full_started
                            full_proxy_risk = risk_for_selection(
                                problem,
                                full.selected,
                                problem.estimated_target_moment,
                            )
                            full_true_risk = risk_for_selection(
                                problem, full.selected, problem.target_moment
                            )
                            rows.append(strategy_record(
                                problem=problem,
                                config=config,
                                strategy_id="full_gram",
                                mode="full_gram",
                                selected=full.selected,
                                full_selected=full.selected,
                                full_proxy_risk=full_proxy_risk,
                                full_true_risk=full_true_risk,
                                transmitted_bytes=packed_bytes,
                                dense_full_bytes=dense_bytes,
                                packed_full_bytes=packed_bytes,
                                elapsed_seconds=full_seconds,
                                decomposition_seconds=decomposition_seconds,
                            ))
                            for rank in valid_ranks:
                                fixed_started = time.monotonic()
                                selected, diagnostics = e1.sketch_a_greedy(
                                    problem,
                                    budget,
                                    rank,
                                    decompositions=decompositions,
                                )
                                fixed_seconds = time.monotonic() - fixed_started
                                rows.append(strategy_record(
                                    problem=problem,
                                    config=config,
                                    strategy_id=f"fixed_l{rank}",
                                    mode="fixed_rank",
                                    selected=selected,
                                    full_selected=full.selected,
                                    full_proxy_risk=full_proxy_risk,
                                    full_true_risk=full_true_risk,
                                    transmitted_bytes=int(diagnostics["sketch_bytes"]),
                                    dense_full_bytes=dense_bytes,
                                    packed_full_bytes=packed_bytes,
                                    elapsed_seconds=fixed_seconds,
                                    decomposition_seconds=decomposition_seconds,
                                    rank_cap=rank,
                                    rank_schedule=[rank],
                                    diagnostics=diagnostics,
                                ))
                            for rank_cap in [
                                rank for rank in valid_ranks if rank < dimension
                            ]:
                                schedule = [
                                    rank for rank in valid_ranks if rank <= rank_cap
                                ]
                                for fallback, mode, prefix in [
                                    (False, "adaptive_capped", "adaptive_cap"),
                                    (
                                        True,
                                        "adaptive_full_fallback",
                                        "adaptive_fallback_cap",
                                    ),
                                ]:
                                    adaptive_started = time.monotonic()
                                    adaptive = adaptive_sketch_target_a(
                                        problem.blocks,
                                        problem.estimated_target_moment,
                                        problem.prior_precision,
                                        budget,
                                        rank_schedule=schedule,
                                        noise_variance=problem.noise_variance,
                                        fallback_to_full=fallback,
                                        decompositions=decompositions,
                                    )
                                    diagnostics = adaptive_interval_diagnostics(
                                        problem, decompositions, adaptive
                                    )
                                    adaptive_seconds = (
                                        time.monotonic() - adaptive_started
                                    )
                                    rows.append(strategy_record(
                                        problem=problem,
                                        config=config,
                                        strategy_id=f"{prefix}_l{rank_cap}",
                                        mode=mode,
                                        selected=adaptive.selected,
                                        full_selected=full.selected,
                                        full_proxy_risk=full_proxy_risk,
                                        full_true_risk=full_true_risk,
                                        transmitted_bytes=adaptive.transmitted_bytes,
                                        dense_full_bytes=dense_bytes,
                                        packed_full_bytes=packed_bytes,
                                        elapsed_seconds=adaptive_seconds,
                                        decomposition_seconds=decomposition_seconds,
                                        rank_cap=rank_cap,
                                        rank_schedule=schedule,
                                        fallback_to_full=fallback,
                                        diagnostics=diagnostics,
                                    ))
                            completed += 1
                            print(
                                f"[{completed}/{total_configurations}] seed={seed} "
                                f"d={dimension} M={candidate_count} K={budget} "
                                f"nT={target_samples} rT={target_rank}",
                                flush=True,
                            )

    e1.write_csv(args.out_dir / "raw.csv", rows)
    summary = summarize(rows, total_configurations)
    elapsed = time.monotonic() - started
    config = {
        "seeds": args.seeds,
        "dimensions": args.dimensions,
        "candidate_counts": args.candidate_counts,
        "budgets": args.budgets,
        "target_samples": args.target_samples,
        "target_rank_fractions": args.target_rank_fractions,
        "source_samples": args.source_samples,
        "sketch_ranks": args.sketch_ranks,
        "mini": args.mini,
        "byte_accounting": {
            "sketch": "nested incremental factor payload; metadata counted once",
            "gate_denominator": "packed symmetric float64 Gram",
            "compatibility_denominator": "dense d-by-d float64 Gram",
        },
    }
    report = {
        "experiment": "E5/H5 synthetic sketch certificate sweep",
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "config": config,
        "counts": {
            "configurations": total_configurations,
            "rows": len(rows),
        },
        "summary": summary,
        "provenance": {
            "git_revision": e1.git_revision(),
            "script_sha256": e1.sha256_file(Path(__file__)),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        key: report[key]
        for key in (
            "experiment",
            "status",
            "created_utc",
            "elapsed_seconds",
            "config",
            "counts",
            "provenance",
        )
    }
    manifest["outputs"] = [
        "config.json",
        "manifest.json",
        "raw.csv",
        "README.md",
        "summary.json",
    ]
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    gate = summary["h5_gate"]
    readme = f"""# E5/H5 Sketch Certificate Sweep

Status: completed synthetic sweep; this is not real-data evidence.

- Configurations: {total_configurations}
- Raw rows: {len(rows)}
- Elapsed seconds: {elapsed:.3f}
- H5 gate: {gate['status']}
- Byte denominator for the gate: packed symmetric float64 full-Gram

`raw.csv` reports deterministic interval coverage, pairwise certificate rate,
final-selection fidelity, and communication ratios. The packed-Gram ratio is
the primary conservative comparison; the dense ratio is retained only for
continuity with the first pilot.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({
        "status": "completed",
        "configurations": total_configurations,
        "rows": len(rows),
        "elapsed_seconds": elapsed,
        "h5_gate": gate["status"],
        "out_dir": str(args.out_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
