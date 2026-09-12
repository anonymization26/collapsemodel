#!/usr/bin/env python3
"""Summarize two-encoder E2b shortlist audits at the target-domain unit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "scripts"))

from metrics.target_conditioned_e2b import (  # noqa: E402
    TEST_AUDIT_SCHEMA,
    E2BArtifactError,
    canonical_json_sha256,
    load_config,
    read_json,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
)
from run_target_conditioned_e2b_shortlist import AUDIT_FIELDS  # noqa: E402


METRICS = (
    "shortlist_reduction",
    "true_top_q_recall",
    "selected_normalized_regret",
    "best_shortlist_normalized_regret",
    "screen_spearman_vs_test_brier",
)
PER_ENCODER_FIELDS = ("encoder", "target_domain", "method", "shortlist_size") + METRICS
PER_UNIT_FIELDS = ("target_domain", "method", "shortlist_size") + METRICS
SUMMARY_FIELDS = (
    "method",
    "shortlist_size",
    "unit_count",
    "mean_shortlist_reduction",
    "mean_true_top_q_recall",
    "recall_ci_low",
    "recall_ci_high",
    "mean_selected_normalized_regret",
    "regret_ci_low",
    "regret_ci_high",
    "mean_best_shortlist_normalized_regret",
    "mean_screen_spearman_vs_test_brier",
    "passes_reduction_gate",
    "passes_recall_gate",
    "passes_regret_gate",
    "passes_all_gates",
)


def _read_audit(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != AUDIT_FIELDS:
            raise E2BArtifactError(f"{path} has unexpected columns")
        return [dict(row) for row in reader]


def _validate_audit(directory: Path, config_path: Path) -> tuple[str, list[dict[str, str]]]:
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema_version") != TEST_AUDIT_SCHEMA:
        raise E2BArtifactError("test audit schema mismatch")
    core = {key: value for key, value in manifest.items() if key != "test_audit_id"}
    if manifest.get("test_audit_id") != canonical_json_sha256(core):
        raise E2BArtifactError("test audit identifier mismatch")
    if manifest.get("config_file_sha256") != sha256_file(config_path):
        raise E2BArtifactError("test audit config mismatch")
    audit_path = directory / "shortlist_audit.csv"
    if manifest.get("audit_file_sha256") != sha256_file(audit_path):
        raise E2BArtifactError("test audit table hash mismatch")
    rows = _read_audit(audit_path)
    if int(manifest.get("audit_row_count", -1)) != len(rows):
        raise E2BArtifactError("test audit row count mismatch")
    encoder = str(manifest["encoder"])
    if any(row["encoder"] != encoder for row in rows):
        raise E2BArtifactError("test audit rows and manifest encoder differ")
    return encoder, rows


def _mean_rows(rows: Sequence[Mapping[str, str]]) -> dict[str, float]:
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in METRICS
    }


def _bootstrap_mean(
    values: Sequence[float], repeats: int, confidence: float, seed: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(repeats, len(array)))
    statistics = array[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(statistics, alpha)), float(
        np.quantile(statistics, 1.0 - alpha)
    )


def summarize(
    config_path: Path,
    audit_dirs: Sequence[Path],
    output_dir: Path,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise E2BArtifactError("refusing to overwrite an E2b summary")
    config = load_config(config_path)
    expected_encoders = [str(value) for value in config["features"]["evaluators"]]
    inputs = []
    all_rows = []
    observed_encoders = []
    for directory in audit_dirs:
        encoder, rows = _validate_audit(directory, config_path)
        observed_encoders.append(encoder)
        all_rows.extend(rows)
        inputs.append(
            {
                "encoder": encoder,
                "manifest_file_sha256": sha256_file(directory / "manifest.json"),
                "audit_file_sha256": sha256_file(directory / "shortlist_audit.csv"),
            }
        )
    if sorted(observed_encoders) != sorted(expected_encoders):
        raise E2BArtifactError("summary does not contain both frozen evaluators")

    grouped_encoder: dict[tuple[str, str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        key = (
            row["encoder"],
            row["target_domain"],
            row["method"],
            int(row["shortlist_size"]),
        )
        grouped_encoder[key].append(row)
    per_encoder = []
    for (encoder, target, method, size), rows in sorted(grouped_encoder.items()):
        expected_repeats = int(config["screen"]["random_repeats"]) if method == "random" else 1
        if len(rows) != expected_repeats:
            raise E2BArtifactError("method repeat coverage is incomplete")
        per_encoder.append(
            {
                "encoder": encoder,
                "target_domain": target,
                "method": method,
                "shortlist_size": size,
                **_mean_rows(rows),
            }
        )

    grouped_units: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in per_encoder:
        grouped_units[(str(row["target_domain"]), str(row["method"]), int(row["shortlist_size"]))].append(row)
    per_unit = []
    for (target, method, size), rows in sorted(grouped_units.items()):
        if len(rows) != len(expected_encoders):
            raise E2BArtifactError("encoder coverage is incomplete within a target unit")
        per_unit.append(
            {
                "target_domain": target,
                "method": method,
                "shortlist_size": size,
                **{
                    metric: float(np.mean([float(row[metric]) for row in rows]))
                    for metric in METRICS
                },
            }
        )

    analysis = config["primary_analysis"]
    repeats = int(analysis["bootstrap_repeats"])
    confidence = float(analysis["confidence_level"])
    gates = analysis["success_gate"]
    grouped_summary: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in per_unit:
        grouped_summary[(str(row["method"]), int(row["shortlist_size"]))].append(row)
    method_summary = []
    for (method, size), rows in sorted(grouped_summary.items()):
        recall = [float(row["true_top_q_recall"]) for row in rows]
        regret = [float(row["selected_normalized_regret"]) for row in rows]
        recall_ci = _bootstrap_mean(recall, repeats, confidence, 20260912 + size)
        regret_ci = _bootstrap_mean(regret, repeats, confidence, 20261912 + size)
        mean_reduction = float(np.mean([float(row["shortlist_reduction"]) for row in rows]))
        mean_recall = float(np.mean(recall))
        mean_regret = float(np.mean(regret))
        pass_reduction = mean_reduction >= float(
            gates["primary_shortlist_reduction_at_least"]
        )
        pass_recall = mean_recall >= float(gates["mean_true_top_q_recall_at_least"])
        pass_regret = mean_regret <= float(
            gates["mean_validation_selected_normalized_regret_at_most"]
        )
        method_summary.append(
            {
                "method": method,
                "shortlist_size": size,
                "unit_count": len(rows),
                "mean_shortlist_reduction": mean_reduction,
                "mean_true_top_q_recall": mean_recall,
                "recall_ci_low": recall_ci[0],
                "recall_ci_high": recall_ci[1],
                "mean_selected_normalized_regret": mean_regret,
                "regret_ci_low": regret_ci[0],
                "regret_ci_high": regret_ci[1],
                "mean_best_shortlist_normalized_regret": float(
                    np.mean([float(row["best_shortlist_normalized_regret"]) for row in rows])
                ),
                "mean_screen_spearman_vs_test_brier": float(
                    np.mean([float(row["screen_spearman_vs_test_brier"]) for row in rows])
                ),
                "passes_reduction_gate": int(pass_reduction),
                "passes_recall_gate": int(pass_recall),
                "passes_regret_gate": int(pass_regret),
                "passes_all_gates": int(pass_reduction and pass_recall and pass_regret),
            }
        )
    primary_method = str(analysis["primary_method"])
    primary_size = int(config["screen"]["primary_shortlist_size"])
    primary = next(
        row
        for row in method_summary
        if row["method"] == primary_method and row["shortlist_size"] == primary_size
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_encoder_path = output_dir / "per_encoder.csv"
    per_unit_path = output_dir / "per_unit.csv"
    method_path = output_dir / "method_summary.csv"
    write_csv_atomic(per_encoder_path, PER_ENCODER_FIELDS, per_encoder)
    write_csv_atomic(per_unit_path, PER_UNIT_FIELDS, per_unit)
    write_csv_atomic(method_path, SUMMARY_FIELDS, method_summary)
    summary: dict[str, object] = {
        "schema_version": "target-conditioned-e2b-summary-v1",
        "config_file_sha256": sha256_file(config_path),
        "inputs": inputs,
        "statistics": {
            "unit": analysis["unit"],
            "encoder_handling": analysis["encoder_handling"],
            "target_unit_count": len(config["dataset"]["domains"]),
            "random_repeats_are_not_independent_units": True,
        },
        "primary_result": primary,
        "h3_passed": bool(primary["passes_all_gates"]),
        "claim_boundary": config["claim_boundary"],
        "per_encoder_file_sha256": sha256_file(per_encoder_path),
        "per_unit_file_sha256": sha256_file(per_unit_path),
        "method_summary_file_sha256": sha256_file(method_path),
    }
    summary["summary_id"] = canonical_json_sha256(summary)
    write_json_atomic(output_dir / "summary.json", summary)
    readme = (
        "# E2b DomainNet 固定成本 shortlist\n\n"
        f"- 主方法：`{primary_method}`\n"
        f"- 主 shortlist：`{primary_size}/455`\n"
        f"- 平均真实 top-10 recall：`{float(primary['mean_true_top_q_recall']):.3f}`\n"
        f"- 平均验证选择测试遗憾：`{float(primary['mean_selected_normalized_regret']):.3%}`\n"
        f"- 平均组合缩减：`{float(primary['mean_shortlist_reduction']):.3%}`\n"
        f"- H3：`{'PASS' if summary['h3_passed'] else 'FAIL'}`\n\n"
        "该结果以六个目标域为统计单位并在域内平均两个评价编码器。它是独立 DomainNet 上的"
        "探索性 E2b，不改变 PACS/Office-Home E2 的 H2/H2b 失败结论。\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.config, args.audit_dir, args.output_dir)
    print(json.dumps({"status": "summarized", "h3_passed": result["h3_passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
