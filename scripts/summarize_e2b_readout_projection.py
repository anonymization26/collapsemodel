#!/usr/bin/env python3
"""Summarize paired supplementary outcomes without treating seeds as datasets."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from run_e2b_readout_projection import check_artifact, load_spec
from metrics.target_conditioned_e2b import E2BArtifactError, load_config, read_json, sha256_file, write_json_atomic


MEASURES = ("recall_at_top_q", "test_loss", "oracle_loss", "absolute_regret",
            "relative_regret", "span_normalized_regret", "absolute_omission", "relative_candidate_span")


def averages(rows):
    result = {}
    for metric in MEASURES:
        values = [r[metric] for r in rows if r[metric] is not None]
        result[metric] = float(np.mean(values)) if values else None
        result[metric + "_defined_count"] = len(values)
    return result


def collapse_random_repeats(rows):
    fields = ("encoder", "projection_seed", "target_domain", "readout", "metric", "shortlist_size")
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in fields) + (row["group"].split(":")[0],)].append(row)
    return [{**dict(zip(fields + ("method",), key)), **averages(values)}
            for key, values in sorted(buckets.items())]


def aggregate(rows, seeds):
    # Equal weight to each target domain after averaging its repeated measures.
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["projection_seed"] in seeds:
            key = tuple(row[k] for k in ("readout", "method", "metric", "shortlist_size"))
            buckets[key][row["target_domain"]].append(row)
    result = []
    for key, domain_rows in sorted(buckets.items()):
        domain_means = [averages(values) for values in domain_rows.values()]
        result.append({**dict(zip(("readout", "method", "metric", "shortlist_size"), key)),
                       "projection_seeds": sorted({r["projection_seed"] for values in domain_rows.values() for r in values}),
                       "requested_projection_seeds": sorted(seeds), "target_domain_count": len(domain_rows),
                       "repeated_task_count": sum(len(v) for v in domain_rows.values()),
                       **averages(domain_means)})
    return result


def compare_r1(root, reference, config):
    base_seed = config["features"]["projection"]["seed"]
    comparison = {}
    for encoder in config["features"]["evaluators"]:
        job = root / f"{encoder}__{base_seed}"
        if not (job / "test-audit/manifest.json").exists():
            continue
        maximum = {m: 0.0 for m in ("brier_score", "nll", "error_rate")}
        observed = {}
        for domain in config["dataset"]["domains"]:
            for name, losses in read_json(job / f"test-audit/{domain}__ridge_losses.json").items():
                observed[(domain, name)] = losses
        path = reference / "test_audit" / encoder / "combinations.csv"
        with path.open(newline="") as stream:
            original = list(csv.DictReader(stream))
        expected_keys = {(r["target_domain"], r["combination"]) for r in original}
        if len(original) != len(observed) or expected_keys != set(observed):
            raise E2BArtifactError("R1 comparison coverage mismatch")
        for row in original:
            current = observed[(row["target_domain"], row["combination"])]
            for metric in maximum:
                before = 1 - float(row["accuracy"]) if metric == "error_rate" else float(row[metric])
                maximum[metric] = max(maximum[metric], abs(current[metric] - before))
        choices = read_json(job / "validate/selections.json")
        choice_map = {(domain, r["group"], r["shortlist_size"]): r["combination"]
                      for domain in config["dataset"]["domains"]
                      for r in choices[f"{domain}__ridge"] if r["metric"] == "brier_score"}
        selection_path = reference / "validation" / encoder / "selections.csv"
        with selection_path.open(newline="") as stream:
            old_choices = list(csv.DictReader(stream))
        changed = []
        for row in old_choices:
            group = f"random:{row['repeat']}" if row["method"] == "random" else row["method"]
            key = (row["target_domain"], group, int(row["shortlist_size"]))
            if choice_map[key] != row["selected_combination"]:
                changed.append({"target_domain": key[0], "group": group, "size": key[2],
                                "old": row["selected_combination"], "new": choice_map[key]})
        comparison[encoder] = {"test_combinations": len(original), "max_absolute_loss_difference": maximum,
                               "validation_selection_count": len(old_choices), "changed_selections": changed,
                               "numerical_match_at_1e-10": max(maximum.values()) <= 1e-10,
                               "reference_hashes": {"combinations": sha256_file(path),
                                                    "selections": sha256_file(selection_path)}}
    return comparison


def summarize(root, output, reference=None, allow_partial=False):
    status = read_json(root / "status.json")
    config = load_config(root / "base_config.json")
    spec = load_spec(root / "protocol.json")
    if status["status"] != "complete" and not allow_partial:
        raise E2BArtifactError("incomplete sweep cannot produce a final summary")
    rows, completed, overlaps = [], [], []
    inputs = {"status.json": sha256_file(root / "status.json"),
              "base_config.json": sha256_file(root / "base_config.json"),
              "protocol.json": sha256_file(root / "protocol.json")}
    for seed in spec["projection_seeds"]:
        for encoder in config["features"]["evaluators"]:
            name = f"{encoder}__{seed}"
            directory = root / name / "test-audit"
            if not (directory / "manifest.json").exists():
                if allow_partial:
                    continue
                raise E2BArtifactError("missing registered task")
            manifest = read_json(directory / "manifest.json")
            context = manifest["context"]
            if (context["encoder"] != encoder or context["projection_seed"] != seed
                    or context["base_config_sha256"] != inputs["base_config.json"]
                    or context["protocol_sha256"] != inputs["protocol.json"]):
                raise E2BArtifactError("summary task identity mismatch")
            check_artifact(directory, context, "test-audit")
            inputs[f"{name}/test-audit/manifest.json"] = sha256_file(directory / "manifest.json")
            observed_rows = read_json(directory / "audit.json")["rows"]
            groups = {m for m in config["screen"]["methods"] if m != "random"}
            groups.update(f"random:{i}" for i in range(config["screen"]["random_repeats"]))
            expected_keys = {(d, r, m, size, group)
                             for d in config["dataset"]["domains"] for r in spec["readouts"]
                             for m in spec["metrics"] for size in config["screen"]["shortlist_sizes"]
                             for group in groups}
            observed_keys = {(r["target_domain"], r["readout"], r["metric"], r["shortlist_size"], r["group"])
                             for r in observed_rows}
            if (len(observed_rows) != len(expected_keys) or observed_keys != expected_keys
                    or any(r["encoder"] != encoder or r["projection_seed"] != seed for r in observed_rows)):
                raise E2BArtifactError("incomplete or duplicate audit rows")
            rows.extend({**r, "relative_candidate_span": (r["worst_loss"] - r["oracle_loss"]) / abs(r["oracle_loss"])
                         if r["oracle_loss"] else None} for r in observed_rows)
            completed.append(name)
            for domain in config["dataset"]["domains"]:
                losses = {r: read_json(directory / f"{domain}__{r}_losses.json") for r in spec["readouts"]}
                for metric in spec["metrics"]:
                    best = {r: set(sorted(losses[r], key=lambda c: (losses[r][c][metric], c))[:10])
                            for r in spec["readouts"]}
                    overlaps.append({"encoder": encoder, "projection_seed": seed,
                                     "target_domain": domain, "metric": metric,
                                     "ridge_logistic_top10_overlap": len(best["ridge"] & best["logistic"]) / 10})
    if not completed:
        raise E2BArtifactError("no complete tasks")
    units = collapse_random_repeats(rows)
    base_seed = config["features"]["projection"]["seed"]
    result = {"status": "complete" if status["status"] == "complete" else "partial-not-final",
              "completed_jobs": completed, "expected_jobs": len(spec["projection_seeds"]) * len(config["features"]["evaluators"]),
              "input_sha256": inputs, "summary_code_sha256": sha256_file(Path(__file__)),
              "unit_rows": units, "base_seed": aggregate(units, {base_seed}),
              "new_seeds": aggregate(units, set(spec["projection_seeds"]) - {base_seed}),
              "per_seed": [row for s in spec["projection_seeds"] for row in aggregate(units, {s})],
              "readout_top10_overlap": overlaps,
              "r1_comparison": compare_r1(root, reference, config) if reference else {},
              "interpretation": spec["evaluation"], "cost_claim": spec["cost_claim"]}
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "summary.json", result)
    for name in ("base_seed", "new_seeds", "per_seed", "unit_rows", "readout_top10_overlap"):
        records = result[name]
        if records:
            with (output / f"{name}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    return {"status": result["status"], "completed_jobs": len(completed),
            "expected_jobs": result["expected_jobs"], "r1_comparison": result["r1_comparison"],
            "primary": [dict(scope=scope, **r) for scope in ("base_seed", "new_seeds")
                        for r in result[scope] if r["metric"] == "brier_score"
                        and r["shortlist_size"] == 227
                        and r["method"] in ("target_a", "second_moment_mmd")]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--r1-reference", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(summarize(args.input_root, args.output_root, args.r1_reference, args.allow_partial), indent=2))
