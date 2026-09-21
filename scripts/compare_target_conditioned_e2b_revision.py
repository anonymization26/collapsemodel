#!/usr/bin/env python3
"""Compare sealed historical and reconstructed E2b outputs without changing them."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from metrics.target_conditioned_e2b import (
    E2BArtifactError, canonical_json_sha256, load_config, read_json,
    sha256_file, write_csv_atomic, write_json_atomic,
)
from run_target_conditioned_e2b_shortlist import (
    METRIC_FIELDS, _read_csv, _spearman, _validate_screen, _validate_validation,
)
from summarize_target_conditioned_e2b import _validate_audit


def load_run(root, encoder, config_path):
    audit_dir = root / "test_audit" / encoder
    actual_encoder, _ = _validate_audit(audit_dir, config_path)
    audit = read_json(audit_dir / "manifest.json")
    screen, screen_rows = _validate_screen(
        root / "screen" / encoder, config_path, audit["candidate_set_id"],
        audit["feature_file_sha256"], encoder)
    validation_dir = root / "validation" / encoder
    validation, _, selections = _validate_validation(
        validation_dir, config_path, screen, audit["feature_file_sha256"], encoder)
    expected = {
        "screen_id": screen["screen_id"], "validation_id": validation["validation_id"],
        "validation_manifest_file_sha256": sha256_file(validation_dir / "manifest.json"),
        "validation_selections_file_sha256": sha256_file(validation_dir / "selections.csv"),
        "combinations_file_sha256": sha256_file(audit_dir / "combinations.csv"),
    }
    if actual_encoder != encoder or any(audit.get(k) != v for k, v in expected.items()):
        raise E2BArtifactError("upstream output identity mismatch")
    rows = _read_csv(audit_dir / "combinations.csv",
                     METRIC_FIELDS + ("test_rank", "in_test_top_q", "normalized_regret"))
    if len(rows) != audit["combination_row_count"]:
        raise E2BArtifactError("test row count mismatch")
    return audit, screen_rows, selections, rows


def indexed(rows, keys):
    mapping = {tuple(row[k] for k in keys): row for row in rows}
    if len(mapping) != len(rows):
        raise E2BArtifactError("duplicate comparison rows")
    return mapping


def compare(config_path, historical, revision, output_dir):
    if output_dir.exists():
        raise E2BArtifactError("refusing to overwrite comparison")
    config = load_config(config_path)
    top_q = int(config["test_audit"]["top_q"])
    units, selection_changes, identities = [], [], []
    for encoder in ("resnet50", "dinov2_b14"):
        old, new = load_run(historical, encoder, config_path), load_run(revision, encoder, config_path)
        for key in ("manifest_id", "candidate_set_id", "config_file_sha256"):
            if old[0][key] != new[0][key]:
                raise E2BArtifactError(f"runs do not share the same {key}")
        old_tasks = {r["target_domain"]: r for r in old[0]["tasks"]}
        new_tasks = {r["target_domain"]: r for r in new[0]["tasks"]}
        if set(old_tasks) != set(new_tasks):
            raise E2BArtifactError("target domain mismatch")
        for domain in old_tasks:
            if old_tasks[domain]["target_test_ids_sha256"] != new_tasks[domain]["target_test_ids_sha256"]:
                raise E2BArtifactError("test sample identity mismatch")
        identities.append({"encoder": encoder, "historical_audit_id": old[0]["test_audit_id"],
                           "revision_audit_id": new[0]["test_audit_id"],
                           "historical_feature_sha256": old[0]["feature_file_sha256"],
                           "revision_feature_sha256": new[0]["feature_file_sha256"]})
        for domain in config["dataset"]["domains"]:
            subsets = []
            for index, keys in ((1, ("method", "repeat", "combination")),
                                (2, ("method", "repeat", "shortlist_size")), (3, ("combination",))):
                a, b = [indexed([row for row in run[index] if row["target_domain"] == domain], keys)
                        for run in (old, new)]
                if a.keys() != b.keys():
                    raise E2BArtifactError("comparison row coverage mismatch")
                subsets.append((a, b))
            screen_a, screen_b = subsets[0]
            selection_a, selection_b = subsets[1]
            test_a, test_b = subsets[2]
            test_keys = sorted(test_a)
            top_a = {key for key in test_a if int(test_a[key]["test_rank"]) <= top_q}
            top_b = {key for key in test_b if int(test_b[key]["test_rank"]) <= top_q}
            changes = [key for key in selection_a
                       if selection_a[key]["selected_combination"] != selection_b[key]["selected_combination"]]
            for method, repeat, size in changes:
                key = (method, repeat, size)
                selection_changes.append({"encoder": encoder, "target_domain": domain,
                                          "method": method, "repeat": repeat, "shortlist_size": size,
                                          "historical_selection": selection_a[key]["selected_combination"],
                                          "revision_selection": selection_b[key]["selected_combination"]})
            units.append({"encoder": encoder, "target_domain": domain,
                          "combination_count": len(test_a), "top_q_overlap": len(top_a & top_b) / top_q,
                          "test_rank_spearman": _spearman([float(test_a[k]["test_rank"]) for k in test_keys],
                                                          [float(test_b[k]["test_rank"]) for k in test_keys]),
                          "max_abs_brier_delta": max(abs(float(test_a[k]["brier_score"]) - float(test_b[k]["brier_score"])) for k in test_a),
                          "max_abs_accuracy_delta": max(abs(float(test_a[k]["accuracy"]) - float(test_b[k]["accuracy"])) for k in test_a),
                          "screen_rank_rows_changed": sum(screen_a[k]["rank"] != screen_b[k]["rank"] for k in screen_a),
                          "screen_rank_row_count": len(screen_a), "validation_selections_changed": len(changes),
                          "validation_selection_count": len(selection_a)})
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv_atomic(output_dir / "per_task.csv", tuple(units[0]), units)
    fields = ("encoder", "target_domain", "method", "repeat", "shortlist_size", "historical_selection", "revision_selection")
    write_csv_atomic(output_dir / "selection_changes.csv", fields, selection_changes)
    core = {"schema_version": "e2b-reconstruction-comparison-v1", "inputs": identities,
            "config_sha256": sha256_file(config_path), "script_sha256": sha256_file(Path(__file__)),
            "per_task_sha256": sha256_file(output_dir / "per_task.csv"),
            "selection_changes_sha256": sha256_file(output_dir / "selection_changes.csv"),
            "task_count": len(units), "changed_validation_selections": len(selection_changes),
            "interpretation": "Same frozen sample/candidate membership; changed device/cache and isolation. Not independent replication or bitwise equivalence."}
    write_json_atomic(output_dir / "manifest.json", {**core, "comparison_id": canonical_json_sha256(core)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "historical", "revision", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    compare(args.config, args.historical, args.revision, args.output_dir)


if __name__ == "__main__":
    main()
