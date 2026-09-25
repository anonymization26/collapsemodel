#!/usr/bin/env python3
"""Check standalone timing decisions and keep feature accounting scopes separate."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

from analyze_e2b_decision_budget import digest, dump_csv, mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from metrics.target_conditioned_e2b import canonical_json_sha256


def aggregate_task_times(records):
    grouped = defaultdict(list)
    fields = ("encoder", "target_domain", "readout", "method", "shortlist_size")
    for row in records:
        grouped[tuple(row[k] for k in fields)].append(row)
    result = []
    for key, values in sorted(grouped.items()):
        if sorted(r["repeat"] for r in values) != [0, 1, 2]:
            raise ValueError("missing or duplicate timing repeats")
        row = {**dict(zip(fields, key)), "repeat_count": 3,
                       "score_seconds": mean([r["score_seconds"] for r in values]),
                       "resident_seconds": mean([r["decision_seconds"] for r in values]),
                       "median_resident_seconds": float(np.median([r["decision_seconds"] for r in values]))}
        for metric in ("brier_score", "nll", "error_rate"):
            field = "test_" + metric
            if field in values[0]:
                row[field] = mean([r[field] for r in values])
        result.append(row)
    return result


def feature_accounting(profile):
    groups = {(r["domain"], r["role"]): r for r in profile["groups"]}
    domains = {domain for domain, _ in groups}
    if len(domains) != 6 or len(groups) != 18 or len(profile["groups"]) != 18:
        raise ValueError("missing or duplicate feature groups")
    result = {}
    for domain in domains:
        for role in ("anchor_pool", "target_selection", "target_validation"):
            row = groups[domain, role]
            if row["count"] != (1024 if role == "target_validation" else 1536) or row["seconds"] <= 0:
                raise ValueError("unexpected feature coverage or elapsed time")
        source = sum(groups[d, "anchor_pool"]["seconds"] for d in domains if d != domain)
        baseline = profile["model_load_seconds"] + source + groups[domain, "target_validation"]["seconds"]
        result[domain] = {"baseline_feature_seconds": baseline,
                          "target_aware_feature_seconds": baseline + groups[domain, "target_selection"]["seconds"]}
    for row in profile["per_target_accounting"]:
        expected = result[row["target_domain"]]
        if any(not np.isclose(row[k], v, rtol=0, atol=1e-10) for k,v in expected.items()):
            raise ValueError("inconsistent additive feature accounting")
    if (len(profile["per_target_accounting"]) != 6
            or {r["target_domain"] for r in profile["per_target_accounting"]} != domains):
        raise ValueError("incomplete target feature accounting")
    return result


def random_union_feature_seconds(profile, rankings, budget):
    blocks = {r["block_id"]: r for r in profile["source_blocks"]}
    if len(blocks) != 18 or any(r["count"] != 512 for r in blocks.values()):
        raise ValueError("incomplete source-block feature profiles")
    counts, seconds = [], []
    for ranking in rankings:
        union = {name for candidate in ranking[:budget] for name in candidate.split("|")}
        counts.append(len(union))
        seconds.append(sum(blocks[n]["seconds"] for n in union))
    return mean(seconds), mean(counts)


def summarize(root, output):
    records, inputs, preparation, feature_costs, profiles, screen_rankings = [], {}, {}, {}, {}, {}
    for encoder in ("resnet50", "dinov2_b14"):
        primary = json.loads((root / "timing" / encoder / "timing.json").read_text())
        sources = [("timing", ("ridge", "logistic"))] if primary.get("ridge_precision") == "float64" else [
            ("timing", ("logistic",)), ("timing_ridge_float64", ("ridge",))]
        preparation[encoder] = {}
        rankings_path = root / encoder / "original/screen/rankings.json"
        rankings = json.loads(rankings_path.read_text())
        screen_rankings[encoder] = rankings
        rank_manifest = json.loads((rankings_path.parent / "manifest.json").read_text())
        if digest(rankings_path) != rank_manifest["files"][rankings_path.name]:
            raise ValueError("frozen rankings hash mismatch")
        inputs[str(rankings_path)] = digest(rankings_path)
        task_losses, test_losses = {}, {}
        for path in (root / encoder / "original/validate").glob("*_losses.json"):
            manifest = json.loads((path.parent / "manifest.json").read_text())
            if digest(path) != manifest["files"][path.name]:
                raise ValueError("frozen validation loss hash mismatch")
            inputs[str(path)] = digest(path)
            task_losses[path.name] = json.loads(path.read_text())
            test_path = path.parent.parent / "test-audit" / path.name
            test_manifest = json.loads((test_path.parent / "manifest.json").read_text())
            if digest(test_path) != test_manifest["files"][test_path.name]:
                raise ValueError("frozen test loss hash mismatch")
            inputs[str(test_path)] = digest(test_path)
            test_losses[path.name] = json.loads(test_path.read_text())
        for directory, readouts in sources:
            timing_path = root / directory / encoder / "timing.json"
            timing = json.loads(timing_path.read_text())
            core = {k: v for k, v in timing.items() if k != "result_id"}
            if timing["status"] != "complete" or canonical_json_sha256(core) != timing["result_id"]:
                raise ValueError("invalid timing seal")
            if timing["protocol_sha256"] != digest(root / "protocol.json"):
                raise ValueError("timing protocol mismatch")
            if timing["cross_method_model_cache"]:
                raise ValueError("cannot report cached models as standalone timing")
            if "ridge" in readouts and timing.get("ridge_precision") != "float64":
                raise ValueError("float64 ridge retiming required")
            inputs[str(timing_path)] = digest(timing_path)
            for name, expected in timing["code_sha256"].items():
                if name == "time_e2b_decisions.py" and timing.get("ridge_precision") != "float64":
                    source = root.parent / "provenance/time_e2b_decisions_initial.py.txt"
                else:
                    canonical = "time_e2b_decisions.py" if name == "time_e2b_decisions_float64.py" else name
                    source = Path(__file__).resolve().parent / canonical
                if digest(source) != expected:
                    raise ValueError("timing source snapshot mismatch")
                inputs[str(source.resolve().relative_to(Path(__file__).resolve().parents[1]))] = expected
            for readout in readouts:
                preparation[encoder][readout] = timing["input_preparation_seconds_all_domains"]
            observed, run_count = [], 0
            for path in sorted(timing_path.parent.glob("*__*.json")):
                run = json.loads(path.read_text())
                if run["readout"] not in readouts:
                    continue
                run_count += 1
                reference = task_losses[f"{run['target_domain']}__{run['readout']}_losses.json"]
                for candidate, losses in run["validation_losses"].items():
                    for metric, value in losses.items():
                        if not np.isclose(value, reference[candidate][metric], rtol=0, atol=1e-10):
                            raise ValueError("timed fit does not reproduce frozen validation loss")
                inputs[str(path)] = digest(path)
                identity = {k: run[k] for k in ("encoder", "target_domain", "readout", "method", "repeat")}
                for checkpoint in run["checkpoints"]:
                    if checkpoint["decision_seconds"] < checkpoint["score_seconds"]:
                        raise ValueError("inconsistent elapsed time")
                    group = f"random:{run['repeat']}" if run["method"] == "random" else run["method"]
                    prefix = rankings[run["target_domain"]][group][:checkpoint["shortlist_size"]]
                    for metric, choice in checkpoint["selections"].items():
                        expected = min(prefix, key=lambda c: (reference[c][metric], c))
                        if choice not in prefix or abs(reference[choice][metric] - reference[expected][metric]) > 1e-10:
                            raise ValueError("timed decision differs from frozen ranking/validation")
                    observed.append({**identity, **checkpoint})
            recorded = [r for r in timing["records"] if r["readout"] in readouts]
            ordered = lambda values: sorted(json.dumps(r, sort_keys=True) for r in values)
            if ordered(observed) != ordered(recorded) or run_count != 72 * len(readouts):
                raise ValueError("incomplete or inconsistent standalone timing files")
            for row in recorded:
                test = test_losses[f"{row['target_domain']}__{row['readout']}_losses.json"]
                for metric, choice in row["selections"].items():
                    row["test_" + metric] = test[choice][metric]
            records += recorded
        profile_path = root / "feature_cost" / encoder / "profile.json"
        profile = json.loads(profile_path.read_text())
        if profile["status"] != "complete" or profile["encoder"] != encoder:
            raise ValueError("incomplete raw-image feature profile")
        profile_source = Path(__file__).with_name("profile_e2b_feature_cost.py")
        if profile["source_code_sha256"] != digest(profile_source):
            raise ValueError("feature profiling source mismatch")
        inputs[str(profile_source.resolve().relative_to(Path(__file__).resolve().parents[1]))] = digest(profile_source)
        inputs[str(profile_path)] = digest(profile_path)
        profiles[encoder] = profile
        if len(profile["source_blocks"]) != 18:
            raise ValueError("missing source-block timing profiles")
        for group in profile["groups"]:
            if group["role"] == "anchor_pool":
                blocks = [r for r in profile["source_blocks"] if r["domain"] == group["domain"]]
                if (len(blocks) != 3 or sum(r["count"] for r in blocks) != group["count"]
                        or not np.isclose(sum(r["seconds"] for r in blocks), group["seconds"], atol=1e-10, rtol=0)):
                    raise ValueError("source block/group accounting mismatch")
        for domain, row in feature_accounting(profile).items():
            feature_costs[encoder, domain] = row
    tasks = aggregate_task_times(records)
    for row in tasks:
        cost = feature_costs[row["encoder"], row["target_domain"]]
        field = "baseline_feature_seconds" if row["method"] in ("random", "exhaustive") else "target_aware_feature_seconds"
        row["feature_seconds"] = cost[field]
        row["source_blocks_charged"] = 15.
        if row["method"] == "random":
            profile = profiles[row["encoder"]]
            rankings = screen_rankings[row["encoder"]][row["target_domain"]]
            source, count = random_union_feature_seconds(profile,
                [rankings[f"random:{i}"] for i in range(3)], row["shortlist_size"])
            validation = next(r["seconds"] for r in profile["groups"]
                              if r["domain"] == row["target_domain"] and r["role"] == "target_validation")
            row["feature_seconds"] = profile["model_load_seconds"] + validation + source
            row["source_blocks_charged"] = count
        row["raw_additive_seconds"] = row["feature_seconds"] + row["resident_seconds"]
    full = {(r["encoder"], r["target_domain"], r["readout"]): r
            for r in tasks if r["method"] == "exhaustive"}
    for row in tasks:
        reference = full[row["encoder"], row["target_domain"], row["readout"]]
        row["resident_vs_full_seconds"] = row["resident_seconds"] - reference["resident_seconds"]
        row["raw_vs_full_seconds"] = row["raw_additive_seconds"] - reference["raw_additive_seconds"]
        for metric in ("brier_score", "nll", "error_rate"):
            row["paired_test_" + metric] = row["test_" + metric] - reference["test_" + metric]
    grouped = defaultdict(list)
    for row in tasks:
        grouped[row["readout"], row["method"], row["shortlist_size"]].append(row)
    curves = []
    for (readout, method, size), values in sorted(grouped.items()):
        if len(values) != 12:
            raise ValueError("missing encoder/domain timing tasks")
        # Two encoder-specific six-target batches; shared preparation is charged once per batch.
        prep = sum(v[readout]["validate"] + (v[readout]["screen"] if method not in ("random", "exhaustive") else 0)
                   for v in preparation.values())
        curves.append({"readout": readout, "method": method, "shortlist_size": size,
                       "task_count": 12,
                       **{field: mean([r[field] for r in values]) for field in
                          ("score_seconds", "resident_seconds", "median_resident_seconds", "feature_seconds", "raw_additive_seconds",
                           "resident_vs_full_seconds", "raw_vs_full_seconds", "test_brier_score", "test_nll", "test_error_rate",
                           "paired_test_brier_score", "paired_test_nll", "paired_test_error_rate")},
                       "cached_two_encoder_batch_seconds": prep + sum(r["resident_seconds"] for r in values),
                       "batch_preparation_seconds": prep})
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "complete", "curves": curves, "task_times": tasks,
              "input_sha256": inputs, "source_sha256": digest(Path(__file__)),
              "preparation_seconds": preparation,
              "aggregation": "arithmetic mean of three matched repetitions per encoder/domain, then equal mean over 12 tasks; timing medians retained as diagnostics",
              "quality_scope": "mean test loss of the exact validation choices made in the three timed repetitions; not the 20-random-ranking science mean",
              "resident_scope": "score-plus-fit-plus-validation; feature loading and projection excluded",
              "cached_batch_scope": "two six-target encoder batches, shared input preparation charged once per encoder",
              "raw_scope": "additive measured image-feature work plus resident decision; not end-to-end replay",
              "random_image_scope": "only the union of blocks used in each timed shortlist; no target-selection images",
              "raw_excludes": ["downloads", "pool construction", "cache serialization", "test audit"],
              "disk_cache": "not flushed", "raw_feature_repeats": 1,
              "ridge_correction": "float32-Gram preliminary timings excluded; separate float64 rerun used"}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    dump_csv(output / "curves.csv", curves)
    dump_csv(output / "task_times.csv", tasks)
    print(json.dumps({"status": "complete", "curves": len(curves), "tasks": len(tasks)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    a = p.parse_args()
    summarize(a.input_root, a.output_root)
