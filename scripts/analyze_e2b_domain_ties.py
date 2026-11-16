#!/usr/bin/env python3
"""Exploratory domain-score tie audit using frozen validation and test losses."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_e2b_decision_budget import aggregate_rows, digest, dump_csv
from run_e2b_decision_challenge import composition
from run_e2b_readout_projection import audit_selection, selections_for
from metrics.target_conditioned_e2b import stable_seed


def shuffle_composition_ties(ranking, domains, seed):
    groups = defaultdict(list)
    for candidate in ranking:
        groups[composition(candidate, domains)].append(candidate)
    signatures = [composition(c, domains) for c in ranking]
    order = list(dict.fromkeys(signatures))
    if signatures != [s for s in order for _ in groups[s]]:
        raise ValueError("domain-score composition strata are not contiguous")
    rng = np.random.default_rng(seed)
    return [c for signature in order
            for c in rng.permutation(sorted(groups[signature])).tolist()]


def analyze(root, verified_summary, output):
    verified = json.loads(verified_summary.read_text())
    for name, expected in verified["input_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError("verified challenge manifest changed")
    spec = json.loads((root / "protocol.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    pending, selections = [], {}
    inputs = {str(verified_summary): digest(verified_summary),
              str(root / "protocol.json"): digest(root / "protocol.json")}
    # Persist every new validation choice before opening any test-loss payload.
    for construction in spec["constructions"]:
        for encoder in ("resnet50", "dinov2_b14"):
            job = root / encoder / construction
            rank_path = job / "screen/rankings.json"
            member_path = job / "screen/candidate_membership.json"
            for path in (rank_path, member_path):
                manifest = json.loads((path.parent / "manifest.json").read_text())
                if digest(path) != manifest["files"][path.name]:
                    raise ValueError("screen payload changed")
                inputs[str(path)] = digest(path)
            ranks = json.loads(rank_path.read_text())
            domains = {n: n.split("__", 1)[0] for n in json.loads(member_path.read_text())}
            for domain, groups in ranks.items():
                rankings = {f"domain_moment_ties:{repeat}": shuffle_composition_ties(
                    groups["domain_moment"], domains,
                    stable_seed(20260922, encoder, construction, domain, repeat))
                    for repeat in range(spec["random_repeats"])}
                for readout in ("ridge", "logistic"):
                    filename = f"{domain}__{readout}_losses.json"
                    path = job / "validate" / filename
                    manifest = json.loads((path.parent / "manifest.json").read_text())
                    if digest(path) != manifest["files"][path.name]:
                        raise ValueError("validation payload changed")
                    inputs[str(path)] = digest(path)
                    validation = json.loads(path.read_text())
                    selected = selections_for(rankings, validation,
                                              spec["shortlist_sizes"], spec["metrics"])
                    identity = {"encoder": encoder, "construction": construction,
                                "target_domain": domain, "readout": readout,
                                "projection_seed": spec["projection_seed"]}
                    key = "__".join((encoder, construction, domain, readout))
                    selections[key] = selected
                    full = {m: min(validation, key=lambda c: (validation[c][m], c))
                            for m in spec["metrics"]}
                    pending.append((job / "test-audit" / filename, identity,
                                    rankings, selected, full))
    selection_path = output / "selections.json"
    selection_path.write_text(json.dumps(selections, sort_keys=True, indent=2) + "\n")
    selection_hash = digest(selection_path)
    raw = defaultdict(list)
    for path, identity, rankings, selected, full in pending:
        manifest = json.loads((path.parent / "manifest.json").read_text())
        if digest(path) != manifest["files"][path.name]:
            raise ValueError("test payload changed")
        inputs[str(path)] = digest(path)
        test = json.loads(path.read_text())
        for r in audit_selection(rankings, test, selected, 10):
            reference = test[full[r["metric"]]][r["metric"]]
            raw[identity["construction"]].append({**identity, **r,
                "method": "domain_moment_ties", "full_validation_test_loss": reference,
                "gap_to_full": r["test_loss"] - reference})
    curves, failures = [], []
    for construction, values in raw.items():
        _, c, f = aggregate_rows(values, spec["absolute_gap_thresholds"])
        curves += [{"construction": construction, **r} for r in c]
        failures += [{"construction": construction, **r} for r in f]
    if digest(selection_path) != selection_hash:
        raise ValueError("validation selections changed during audit")
    result = {"status": "complete", "analysis": "post-hoc-randomized-domain-score-ties",
              "interpretation": "supplementary-to-original-lexicographic-tie-protocol",
              "random_repeats": spec["random_repeats"], "curves": curves,
              "failure_rates": failures, "input_sha256": inputs,
              "selections_sha256": selection_hash, "source_sha256": digest(Path(__file__))}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    dump_csv(output / "curves.csv", curves)
    dump_csv(output / "failure_rates.csv", failures)
    print(json.dumps({"status": "complete", "curves": len(curves)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "verified-summary", "output-root"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    analyze(a.input_root, a.verified_summary, a.output_root)
