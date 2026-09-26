#!/usr/bin/env python3
"""Audit A-opt/MMD complementarity using frozen rankings and recorded losses."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import fmean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from metrics.target_conditioned_e2b import (  # noqa: E402
    canonical_json_sha256, read_json, sha256_file, write_json_atomic,
)


METHODS = (
    "mmd", "a_opt", "union", "mmd_matched_union_size",
    "interleave_mmd_first", "interleave_a_first", "mmd_2s_then_a",
)
IDENTITY = ("scope", "encoder", "projection_seed", "target_domain", "readout", "metric")
RESCUE_FIELDS = IDENTITY + (
    "shortlist_size", "direction", "candidate", "validation_loss", "test_loss",
    "selected_by_union", "test_delta_to_baseline_choice", "validation_delta_to_baseline_choice",
)
GROUP = ("scope", "readout", "metric", "shortlist_size")
PAIR_MEASURES = (
    "overlap_count", "union_size", "jaccard", "mmd_missed_top_q", "a_missed_top_q",
    "a_rescued_top_q", "mmd_rescued_top_q", "a_rescued_fractional_top_q",
    "mmd_rescued_fractional_top_q", "omission_reduction_from_union",
    "a_only_better_than_mmd_choice", "top_q_boundary_ties",
)
PAIR_EVENTS = (
    "has_a_rescue", "has_mmd_rescue", "a_rescues_any_oracle", "mmd_rescues_any_oracle",
    "union_selects_a_only", "union_better_than_mmd", "union_worse_than_mmd",
    "union_better_than_matched_mmd", "union_worse_than_matched_mmd",
    "a_rescue_reaches_final_improvement",
)
CHOICE_MEASURES = (
    "evaluated_candidates", "validation_loss", "test_loss", "delta_vs_mmd",
    "delta_vs_matched_mmd", "gap_to_full_validation", "recall_at_top_q",
    "fractional_recall_at_top_q", "absolute_omission",
)
CHOICE_EVENTS = (
    "better_than_mmd", "worse_than_mmd", "better_than_matched_mmd",
    "worse_than_matched_mmd", "selected_changed_from_mmd", "selected_outside_mmd",
    "selected_from_a_only", "any_test_oracle_retained",
)


def checked_manifest(directory, context, stage, inputs):
    path = directory / "manifest.json"
    artifact = read_json(path)
    core = {k: v for k, v in artifact.items() if k != "artifact_id"}
    if (artifact.get("artifact_id") != canonical_json_sha256(core)
            or artifact.get("context") != context or artifact.get("stage") != stage
            or artifact.get("status") != "complete"):
        raise ValueError(f"invalid {stage} manifest: {directory}")
    inputs[str(path.relative_to(ROOT))] = sha256_file(path)
    return artifact


def checked_payload(directory, manifest, name, inputs):
    path = directory / name
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("invalid payload path")
    digest = sha256_file(path)
    if manifest["files"].get(name) != digest:
        raise ValueError(f"payload hash mismatch: {path}")
    inputs[str(path.relative_to(ROOT))] = digest
    return read_json(path)


def validate_task(a_ranking, m_ranking, validation, test, metrics, expected):
    universe = set(a_ranking)
    if (len(a_ranking) != expected or len(universe) != expected
            or len(m_ranking) != expected or set(m_ranking) != universe
            or set(validation) != universe or set(test) != universe):
        raise ValueError("candidate coverage or duplicate ranking mismatch")
    for losses in (validation, test):
        for values in losses.values():
            if any(not math.isfinite(values[m]) or values[m] < 0 for m in metrics):
                raise ValueError("invalid recorded loss")


def interleave(first, second, size):
    chosen, seen = [], set()
    for pair in zip(first, second):
        for candidate in pair:
            if candidate not in seen:
                chosen.append(candidate)
                seen.add(candidate)
                if len(chosen) == size:
                    return chosen
    raise ValueError("insufficient interleaving candidates")


def make_shortlists(a_ranking, m_ranking, size, multiplier=2):
    if (size < 1 or size > len(a_ranking) or multiplier < 1
            or len(set(a_ranking)) != len(a_ranking)
            or len(set(m_ranking)) != len(m_ranking)
            or set(a_ranking) != set(m_ranking)):
        raise ValueError("invalid rankings or budget")
    a, m = a_ranking[:size], m_ranking[:size]
    union = sorted(set(a) | set(m))
    eligible = set(m_ranking[:min(multiplier * size, len(m_ranking))])
    return {
        "mmd": m, "a_opt": a, "union": union,
        "mmd_matched_union_size": m_ranking[:len(union)],
        "interleave_mmd_first": interleave(m_ranking, a_ranking, size),
        "interleave_a_first": interleave(a_ranking, m_ranking, size),
        "mmd_2s_then_a": [c for c in a_ranking if c in eligible][:size],
    }


def choose(candidates, validation, metric):
    return min(candidates, key=lambda c: (validation[c][metric], c))


def top_membership(test, metric, top_q):
    ordered = sorted(test, key=lambda c: (test[c][metric], c))
    q = min(top_q, len(ordered))
    if q < 1:
        raise ValueError("top_q must be positive")
    cutoff = test[ordered[q - 1]][metric]
    strict = sum(v[metric] < cutoff for v in test.values())
    tied = sum(v[metric] == cutoff for v in test.values())
    weights = {c: 1.0 if v[metric] < cutoff else (q - strict) / tied
               if v[metric] == cutoff else 0.0 for c, v in test.items()}
    minimum = test[ordered[0]][metric]
    oracles = {c for c, v in test.items() if v[metric] == minimum}
    return set(ordered[:q]), weights, oracles, tied


def evaluate_task(a_ranking, m_ranking, validation, test, metric, size,
                  top_q=10, tolerance=1e-12):
    validate_task(a_ranking, m_ranking, validation, test, [metric], len(a_ranking))
    # List construction and validation selection do not use test values.
    lists = make_shortlists(a_ranking, m_ranking, size)
    selected = {method: choose(candidates, validation, metric)
                for method, candidates in lists.items()}
    full = choose(a_ranking, validation, metric)
    m, a = set(lists["mmd"]), set(lists["a_opt"])
    union, a_only, m_only = m | a, a - m, m - a
    top, weights, oracles, tied = top_membership(test, metric, top_q)
    value = lambda c: test[c][metric]
    m_loss = value(selected["mmd"])
    matched_loss = value(selected["mmd_matched_union_size"])
    union_delta = value(selected["union"]) - m_loss
    union_matched_delta = value(selected["union"]) - matched_loss
    pair = {
        "shortlist_size": size, "overlap_count": len(a & m), "union_size": len(union),
        "jaccard": len(a & m) / len(union),
        "mmd_missed_top_q": len(top - m), "a_missed_top_q": len(top - a),
        "a_rescued_top_q": len(a_only & top), "mmd_rescued_top_q": len(m_only & top),
        "a_rescued_fractional_top_q": sum(weights[c] for c in a_only),
        "mmd_rescued_fractional_top_q": sum(weights[c] for c in m_only),
        "omission_reduction_from_union": min(map(value, m)) - min(map(value, union)),
        "a_only_better_than_mmd_choice": sum(value(c) < m_loss - tolerance for c in a_only),
        "top_q_boundary_ties": tied,
        "has_a_rescue": bool(a_only & top), "has_mmd_rescue": bool(m_only & top),
        "a_rescues_any_oracle": not bool(m & oracles) and bool(a_only & oracles),
        "mmd_rescues_any_oracle": not bool(a & oracles) and bool(m_only & oracles),
        "union_selects_a_only": selected["union"] in a_only,
        "union_better_than_mmd": union_delta < -tolerance,
        "union_worse_than_mmd": union_delta > tolerance,
        "union_better_than_matched_mmd": union_matched_delta < -tolerance,
        "union_worse_than_matched_mmd": union_matched_delta > tolerance,
        "a_rescue_reaches_final_improvement": selected["union"] in (a_only & top)
            and union_delta < -tolerance,
        "a_only_top_q": sorted(a_only & top), "mmd_only_top_q": sorted(m_only & top),
        "mmd_choice": selected["mmd"], "union_choice": selected["union"],
    }
    choices = []
    for method, candidates in lists.items():
        choice = selected[method]
        loss = value(choice)
        delta, matched_delta = loss - m_loss, loss - matched_loss
        choices.append({
            "shortlist_size": size, "method": method, "evaluated_candidates": len(candidates),
            "selection": choice, "validation_loss": validation[choice][metric], "test_loss": loss,
            "delta_vs_mmd": delta, "delta_vs_matched_mmd": matched_delta,
            "gap_to_full_validation": loss - value(full),
            "recall_at_top_q": len(set(candidates) & top) / len(top),
            "fractional_recall_at_top_q": sum(weights[c] for c in candidates) / len(top),
            "absolute_omission": min(map(value, candidates)) - min(map(value, oracles)),
            "better_than_mmd": delta < -tolerance, "worse_than_mmd": delta > tolerance,
            "better_than_matched_mmd": matched_delta < -tolerance,
            "worse_than_matched_mmd": matched_delta > tolerance,
            "selected_changed_from_mmd": choice != selected["mmd"],
            "selected_outside_mmd": choice not in m, "selected_from_a_only": choice in a_only,
            "any_test_oracle_retained": bool(set(candidates) & oracles),
        })
    rescued = []
    for direction, good, missed_list in (
        ("a_rescues_mmd", a_only & top, "mmd"),
        ("mmd_rescues_a", m_only & top, "a_opt"),
    ):
        baseline = selected[missed_list]
        for c in sorted(good):
            rescued.append({
                "shortlist_size": size, "direction": direction, "candidate": c,
                "validation_loss": validation[c][metric], "test_loss": value(c),
                "selected_by_union": c == selected["union"],
                "test_delta_to_baseline_choice": value(c) - value(baseline),
                "validation_delta_to_baseline_choice": validation[c][metric] - validation[baseline][metric],
            })
    return pair, choices, rescued


def aggregate(rows, keys, measures, events, tolerance=1e-12):
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        buckets[tuple(row[k] for k in keys)][row["target_domain"]].append(row)
    domains, summaries = [], []
    for key, groups in sorted(buckets.items()):
        identity = dict(zip(keys, key))
        domain_rows = []
        for domain, values in sorted(groups.items()):
            row = {**identity, "target_domain": domain, "repeated_task_count": len(values)}
            row.update({m + "_mean": fmean(r[m] for r in values) for m in measures})
            for event in events:
                row[event + "_tasks"] = sum(r[event] for r in values)
                row[event + "_rate"] = fmean(r[event] for r in values)
            domain_rows.append(row)
        summary = {**identity, "domain_count": len(groups),
                   "repeated_task_count": sum(len(v) for v in groups.values())}
        summary.update({m + "_mean": fmean(r[m + "_mean"] for r in domain_rows) for m in measures})
        for event in events:
            summary[event + "_tasks"] = sum(r[event + "_tasks"] for r in domain_rows)
            summary[event + "_rate"] = fmean(r[event + "_rate"] for r in domain_rows)
        for measure in ("delta_vs_mmd", "delta_vs_matched_mmd"):
            if measure in measures:
                summary[measure + "_domains_better"] = sum(r[measure + "_mean"] < -tolerance for r in domain_rows)
                summary[measure + "_domains_worse"] = sum(r[measure + "_mean"] > tolerance for r in domain_rows)
        domains.extend(domain_rows)
        summaries.append(summary)
    return domains, summaries


def dump_csv(path, rows, fields=None):
    if not rows and fields is None:
        raise ValueError(f"empty report: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields if fields is not None else list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})


def analyze(protocol, output):
    if output.exists():
        raise ValueError("output already exists; use a new directory")
    spec = read_json(protocol)
    if (spec["schema_version"] != "e2b-aopt-mmd-complementarity-v1"
            or tuple(spec["methods"]) != METHODS or spec["cascade_multiplier"] != 2):
        raise ValueError("unsupported audit protocol")
    root = ROOT / spec["input_root"]
    config, original = read_json(root / "base_config.json"), read_json(root / "protocol.json")
    if (read_json(root / "status.json")["status"] != "complete"
            or spec["shortlist_sizes"] != config["screen"]["shortlist_sizes"]
            or spec["metrics"] != original["metrics"] or spec["readouts"] != original["readouts"]):
        raise ValueError("incomplete or incompatible original study")
    expected = config["combinations"]["expected_combinations_per_target"]
    base_seed = config["features"]["projection"]["seed"]
    inputs = {str(p.relative_to(ROOT)): sha256_file(p)
              for p in (protocol, root / "base_config.json", root / "protocol.json", root / "status.json")}
    pairs, choices, rescued, missing_models = [], [], [], []
    original_checks = 0
    for seed in original["projection_seeds"]:
        for encoder in config["features"]["evaluators"]:
            job = root / f"{encoder}__{seed}"
            context = read_json(job / "screen/manifest.json")["context"]
            if (context["encoder"] != encoder or context["projection_seed"] != seed
                    or context["base_config_sha256"] != sha256_file(root / "base_config.json")
                    or context["protocol_sha256"] != sha256_file(root / "protocol.json")):
                raise ValueError("job identity mismatch")
            manifests = {stage: checked_manifest(job / stage, context, stage, inputs)
                         for stage in ("screen", "validate", "test-audit")}
            screen, valid, audit = (manifests[s] for s in ("screen", "validate", "test-audit"))
            if (screen["upstream"] != {}
                    or valid["upstream"] != {"screen": screen["artifact_id"]}
                    or audit["upstream"] != {"screen": screen["artifact_id"], "validate": valid["artifact_id"]}
                    or not screen["lineage"] == valid["lineage"] == audit["lineage"]
                    or any(screen["access"].values()) or valid["access"]["target_test"]):
                raise ValueError("stage lineage or information-access mismatch")
            # Model arrays are not needed for replaying recorded decisions.
            missing_models.extend(str((job / "validate" / n).relative_to(ROOT))
                                  for n in valid["files"] if n.endswith(".npz")
                                  and not (job / "validate" / n).exists())
            rankings = checked_payload(job / "screen", screen, "rankings.json", inputs)
            saved_choices = checked_payload(job / "validate", valid, "selections.json", inputs)
            saved_rows = checked_payload(job / "test-audit", audit, "audit.json", inputs)["rows"]
            old_audit = {}
            for r in saved_rows:
                key = tuple(r[k] for k in ("target_domain", "readout", "metric", "shortlist_size", "group"))
                if key in old_audit or r["encoder"] != encoder or r["projection_seed"] != seed:
                    raise ValueError("duplicate or misidentified original audit row")
                old_audit[key] = r
            if set(rankings) != set(config["dataset"]["domains"]):
                raise ValueError("domain coverage mismatch")
            for domain in config["dataset"]["domains"]:
                a, m = rankings[domain]["target_a"], rankings[domain]["second_moment_mmd"]
                for readout in spec["readouts"]:
                    filename = f"{domain}__{readout}_losses.json"
                    validation = checked_payload(job / "validate", valid, filename, inputs)
                    test = checked_payload(job / "test-audit", audit, filename, inputs)
                    validate_task(a, m, validation, test, spec["metrics"], expected)
                    saved = saved_choices[f"{domain}__{readout}"]
                    old_selection = {(r["group"], r["shortlist_size"], r["metric"]): r for r in saved}
                    if len(old_selection) != len(saved):
                        raise ValueError("duplicate original selection")
                    for metric in spec["metrics"]:
                        identity = dict(zip(IDENTITY, (
                            "base_seed_control" if seed == base_seed else "new_seeds",
                            encoder, seed, domain, readout, metric)))
                        for size in spec["shortlist_sizes"]:
                            pair, selected, events = evaluate_task(a, m, validation, test, metric, size,
                                                                  spec["top_q"], spec["comparison_absolute_tolerance"])
                            for row in selected:
                                if row["method"] not in ("mmd", "a_opt"):
                                    continue
                                group = "target_a" if row["method"] == "a_opt" else "second_moment_mmd"
                                recorded = old_selection[group, size, metric]
                                audited = old_audit[domain, readout, metric, size, group]
                                if (recorded["combination"] != row["selection"]
                                        or audited["combination"] != row["selection"]
                                        or abs(recorded["validation_loss"] - row["validation_loss"]) > 1e-12
                                        or any(abs(audited[k] - row[k]) > 1e-12
                                               for k in ("test_loss", "recall_at_top_q", "absolute_omission"))):
                                    raise ValueError("original A-opt/MMD replay mismatch")
                                original_checks += 1
                            pairs.append({**identity, **pair})
                            choices.extend({**identity, **r} for r in selected)
                            rescued.extend({**identity, **r} for r in events)
    pair_domains, pair_summary = aggregate(pairs, GROUP, PAIR_MEASURES, PAIR_EVENTS)
    choice_domains, choice_summary = aggregate(choices, GROUP + ("method",), CHOICE_MEASURES, CHOICE_EVENTS)
    expected_pairs = (len(original["projection_seeds"]) * len(config["features"]["evaluators"])
                      * len(config["dataset"]["domains"]) * len(spec["readouts"])
                      * len(spec["metrics"]) * len(spec["shortlist_sizes"]))
    if len(pairs) != expected_pairs or original_checks != 2 * expected_pairs:
        raise ValueError("incomplete replay")
    output.mkdir(parents=True, exist_ok=False)
    tables = {"pair_tasks": pairs, "selection_tasks": choices, "rescue_candidates": rescued,
              "pair_domains": pair_domains, "pair_summary": pair_summary,
              "selection_domains": choice_domains, "selection_summary": choice_summary}
    for name, rows in tables.items():
        dump_csv(output / f"{name}.csv", rows, RESCUE_FIELDS if name == "rescue_candidates" else None)
    manifest = {
        "status": "complete", "protocol": spec, "input_sha256": inputs,
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "original_aopt_mmd_decisions_verified": original_checks,
        "pair_task_rows": len(pairs), "selection_task_rows": len(choices),
        "rescue_candidate_rows": len(rescued), "model_arrays_not_present_and_not_needed": missing_models,
        "verification_scope": "artifact-identities-lineage-and-all-consumed-ranking-selection-loss-files; no-model-refit-or-npz-verification",
        "outputs": {f"{name}.csv": sha256_file(output / f"{name}.csv") for name in tables},
    }
    write_json_atomic(output / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("status", "original_aopt_mmd_decisions_verified",
                                               "pair_task_rows", "selection_task_rows", "rescue_candidate_rows")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "code/configs/target_conditioned_e2b/complementarity_audit_v1.json")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.protocol.resolve(), args.output_root.resolve())
