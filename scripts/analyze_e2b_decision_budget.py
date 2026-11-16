#!/usr/bin/env python3
"""Post-hoc decision curves with a validation-selected exhaustive reference."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return float(np.mean(values))


def dump_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_reference(validation, test, metric):
    if set(validation) != set(test) or not validation:
        raise ValueError("candidate coverage mismatch")
    choice = min(validation, key=lambda c: (validation[c][metric], c))
    return choice, test[choice][metric]


def aggregate_rows(rows, thresholds):
    fields = ("encoder", "projection_seed", "target_domain", "readout", "metric", "shortlist_size", "method")
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in fields)].append(row)
    units = []
    for key, values in sorted(grouped.items()):
        unit = dict(zip(fields, key))
        for field in ("test_loss", "oracle_loss", "full_validation_test_loss", "gap_to_full",
                      "absolute_omission", "recall_at_top_q"):
            unit[field] = mean([r[field] for r in values])
        unit["random_repeat_count"] = len(values)
        unit["failure_rates"] = {str(t): mean([r["gap_to_full"] > t for r in values])
                                  for t in thresholds[unit["metric"]]}
        units.append(unit)
    grouped = defaultdict(lambda: defaultdict(list))
    for u in units:
        grouped[tuple(u[k] for k in ("readout", "metric", "method", "shortlist_size"))][u["target_domain"]].append(u)
    curves, failures = [], []
    for key, domains in sorted(grouped.items()):
        identity = dict(zip(("readout", "metric", "method", "shortlist_size"), key))
        row = {**identity, "domain_count": len(domains), "repeated_task_count": sum(map(len, domains.values()))}
        for field in ("test_loss", "oracle_loss", "full_validation_test_loss", "gap_to_full",
                      "absolute_omission", "recall_at_top_q"):
            row[field] = mean([mean([u[field] for u in values]) for values in domains.values()])
        row["worst_repeated_task_gap"] = max(u["gap_to_full"] for values in domains.values() for u in values)
        curves.append(row)
        for t in thresholds[identity["metric"]]:
            rates = [mean([u["failure_rates"][str(t)] for u in values]) for values in domains.values()]
            failures.append({**identity, "absolute_gap_threshold": t, "failure_rate": mean(rates),
                             "worst_domain_failure_rate": max(rates), "domain_count": len(domains)})
    return units, curves, failures


def paired_comparisons(units):
    index = {tuple(u[k] for k in ("encoder", "projection_seed", "target_domain", "readout", "metric", "method", "shortlist_size")): u
             for u in units}
    domains = defaultdict(list)
    for key, u in index.items():
        if u["method"] not in ("target_a", "second_moment_mmd"):
            continue
        for comparator, budget in dict.fromkeys((("random", u["shortlist_size"]), ("random", 227), ("exhaustive", 455))):
            other = index[key[:5] + (comparator, budget)]
            pair = (u["readout"], u["metric"], u["method"], u["shortlist_size"], comparator, budget, u["target_domain"])
            domains[pair].append(u["test_loss"] - other["test_loss"])
    fields = ("readout", "metric", "method", "shortlist_size", "comparator", "comparator_budget", "target_domain")
    per_domain = [{**dict(zip(fields, key)), "paired_loss_difference": mean(values)}
                  for key, values in sorted(domains.items())]
    groups = defaultdict(list)
    for r in per_domain:
        groups[tuple(r[k] for k in fields[:-1])].append(r["paired_loss_difference"])
    rng = np.random.default_rng(20260922)
    summaries = []
    for key, values in sorted(groups.items()):
        values = np.asarray(values)
        draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
        summaries.append({**dict(zip(fields[:-1], key)), "paired_loss_difference": mean(values),
                          "descriptive_domain_bootstrap_low": float(np.quantile(draws, .025)),
                          "descriptive_domain_bootstrap_high": float(np.quantile(draws, .975)),
                          "domains_better": int(np.sum(values < -1e-12)),
                          "domains_worse": int(np.sum(values > 1e-12)), "domain_count": len(values)})
    return per_domain, summaries


def analyze(root, protocol, output):
    spec = json.loads(protocol.read_text())
    inputs = {str(protocol): digest(protocol)}
    records = []
    jobs = sorted(p for p in root.glob("*__*/test-audit") if not p.parent.name.endswith("__20260911"))
    if len(jobs) != 8:
        raise ValueError("expected two encoders times four new seeds")
    for audit_dir in jobs:
        manifest = json.loads((audit_dir / "manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError("incomplete audit")
        for name, expected in manifest["files"].items():
            if digest(audit_dir / name) != expected:
                raise ValueError("audit hash mismatch")
        inputs[str(audit_dir / "manifest.json")] = digest(audit_dir / "manifest.json")
        rows = json.loads((audit_dir / "audit.json").read_text())["rows"]
        validate_dir = audit_dir.parent / "validate"
        vmanifest = json.loads((validate_dir / "manifest.json").read_text())
        inputs[str(validate_dir / "manifest.json")] = digest(validate_dir / "manifest.json")
        tasks = sorted({(r["target_domain"], r["readout"], r["metric"]) for r in rows})
        for domain, readout, metric in tasks:
            filename = f"{domain}__{readout}_losses.json"
            path = validate_dir / filename
            if digest(path) != vmanifest["files"][filename]:
                raise ValueError("validation loss hash mismatch")
            inputs[str(path)] = digest(path)
            validation = json.loads(path.read_text())
            test = json.loads((audit_dir / filename).read_text())
            _, reference = select_reference(validation, test, metric)
            task_rows = [r for r in rows if (r["target_domain"], r["readout"], r["metric"]) == (domain, readout, metric)]
            for r in task_rows:
                if r["group"].split(":")[0] not in ("target_a", "second_moment_mmd", "random"):
                    continue
                records.append({**r, "method": r["group"].split(":")[0],
                                "full_validation_test_loss": reference, "gap_to_full": r["test_loss"] - reference})
            r = task_rows[0]
            records.append({**{k:r[k] for k in ("encoder", "projection_seed", "target_domain", "readout", "metric")},
                            "method": "exhaustive", "shortlist_size": len(test), "test_loss": reference,
                            "oracle_loss": min(v[metric] for v in test.values()),
                            "full_validation_test_loss": reference, "gap_to_full": 0.,
                            "absolute_omission": 0., "recall_at_top_q": 1.})
    units, curves, failures = aggregate_rows(records, spec["absolute_gap_thresholds"])
    domains, pairs = paired_comparisons(units)
    output.mkdir(parents=True, exist_ok=False)
    payload = {"status": "complete", "analysis": "post-hoc-new-seed-budget-audit",
               "threshold_status": spec["threshold_interpretation"],
               "statistical_scope": "six-domain-descriptive-repeated-measures-not-independent-datasets",
               "time_status": "candidate-count-only-no-measured-time-in-this-analysis",
               "input_sha256": inputs, "analysis_code_sha256": digest(Path(__file__)),
               "curves": curves, "failure_rates": failures, "paired_domains": domains,
               "paired_summary": pairs, "unit_rows": units}
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    for name, values in (("curves", curves), ("failure_rates", failures), ("paired_domains", domains), ("paired_summary", pairs)):
        dump_csv(output / f"{name}.csv", values)
    print(json.dumps({"curves": len(curves), "repeated_units": len(units), "paired_summaries": len(pairs)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "protocol", "output-root"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    analyze(a.input_root, a.protocol, a.output_root)
