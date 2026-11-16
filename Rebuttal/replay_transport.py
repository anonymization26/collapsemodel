"""Replay DomainNet logistic outcomes for frozen fixed-pool transport rankings."""

import argparse
import csv
import hashlib
import json
from math import isfinite
from pathlib import Path
from statistics import mean


ARCHIVE = Path(__file__).resolve().parent / (
    "runs/replay-001/package/results/target_conditioned/"
    "revision_20260917/revision_results/e2b_readout_projection_v1"
)
BUDGETS = (50, 227)
DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
ENCODERS = ("resnet50", "dinov2_b14")
TRANSPORT = ("sliced_w2", "whitened_sliced_w2")
BASELINES = ("target_a", "second_moment_mmd")
METRICS = ("test_brier", "full_test_brier", "delta_brier")


def checked_json(folder, manifest, name):
    payload = (folder / name).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != manifest["files"][name]:
        raise ValueError(f"Archived hash mismatch: {folder / name}")
    return json.loads(payload)


def unit_decisions(archive, encoder, domain, path):
    folder = archive / f"{encoder}__20260911"
    screen, validate, test = (
        json.loads((folder / stage / "manifest.json").read_text())
        for stage in ("screen", "validate", "test-audit")
    )
    if not (
        screen["lineage"] == validate["lineage"] == test["lineage"]
        and validate["upstream"]["screen"] == test["upstream"]["screen"] == screen["artifact_id"]
        and test["upstream"]["validate"] == validate["artifact_id"]
    ):
        raise ValueError(f"Archive lineage mismatch for {encoder}/{domain}")

    original = checked_json(folder / "screen", screen, "rankings.json")[domain]
    loss_name = f"{domain}__logistic_losses.json"
    validation = checked_json(folder / "validate", validate, loss_name)
    candidates = set(validation)
    if len(candidates) != 455 or any(
        not isfinite(row["brier_score"]) for row in validation.values()
    ):
        raise ValueError(f"Invalid validation losses for {encoder}/{domain}")

    payload = path.read_bytes()
    result = json.loads(payload)
    for key, value in (("encoder", encoder), ("domain", domain)):
        if key in result and result[key] != value:
            raise ValueError(f"Screen output {path} does not match {key} {value}")
    ranks = {method: [result["transport"][method]] for method in TRANSPORT}
    ranks.update({method: [original[method]] for method in BASELINES})
    ranks["random"] = [original[f"random:{i}"] for i in range(20)]
    if not all(
        len(order) == 455 and set(order) == candidates
        for orders in ranks.values()
        for order in orders
    ):
        raise ValueError(f"Candidate IDs or multiplicity differ for {encoder}/{domain}")

    def winner(ids):
        return min(ids, key=lambda candidate: (validation[candidate]["brier_score"], candidate))

    full = winner(validation)
    decisions = [
        dict(encoder=encoder, domain=domain, budget=budget, method=method,
             repeat=repeat, winner=winner(order[:budget]), full_winner=full)
        for budget in BUDGETS
        for method, orders in ranks.items()
        for repeat, order in enumerate(orders)
    ]
    decisions.append(dict(encoder=encoder, domain=domain, budget=455, method="full",
                          repeat=0, winner=full, full_winner=full))
    metadata = dict(encoder=encoder, domain=domain, screen_file=str(path),
                    screen_sha256=hashlib.sha256(payload).hexdigest(),
                    archive_lineage=screen["lineage"],
                    timing=result.get("timing"), exact=result.get("exact"))
    return decisions, metadata, (folder / "test-audit", test, loss_name, candidates)


def aggregate(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    return [
        dict(zip(keys, key), **{metric: mean(row[metric] for row in group) for metric in METRICS})
        for key, group in groups.items()
    ]


def write_csv(folder, name, rows):
    with (folder / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", nargs=3, action="append", required=True,
                        metavar=("ENCODER", "DOMAIN", "SCREEN_JSON"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    keys = [(encoder, domain) for encoder, domain, _ in args.unit]
    if len(set(keys)) != len(keys):
        parser.error("Each encoder/domain unit must appear only once")
    if any(encoder not in ENCODERS or domain not in DOMAINS for encoder, domain in keys):
        parser.error("Unknown encoder/domain unit")
    encoder_sets = [
        {encoder for encoder, observed in keys if observed == domain}
        for domain in DOMAINS if any(observed == domain for _, observed in keys)
    ]
    if any(encoders != encoder_sets[0] for encoders in encoder_sets):
        parser.error("Each included domain must have the same encoder set")

    decisions, screen_metadata, audit_inputs = [], [], {}
    for encoder, domain, path in args.unit:
        rows, metadata, inputs = unit_decisions(args.archive, encoder, domain, Path(path))
        decisions.extend(rows)
        screen_metadata.append(metadata)
        audit_inputs[encoder, domain] = inputs
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "frozen_decisions.json").write_text(json.dumps(decisions, indent=2) + "\n")
    (output / "screen_metadata.json").write_text(json.dumps(screen_metadata, indent=2) + "\n")
    print(f"PASS: {len(keys)} units matched the archived lineage, ranking/validation hashes "
          "and all 455 candidate IDs. Winners written before opening test outcomes.", flush=True)

    losses = {}
    for key, (folder, manifest, name, candidates) in audit_inputs.items():
        test = checked_json(folder, manifest, name)
        if set(test) != candidates or any(not isfinite(row["brier_score"]) for row in test.values()):
            raise ValueError(f"Invalid test outcomes for {key}")
        losses[key] = test
    for row in decisions:
        test = losses[row["encoder"], row["domain"]]
        row["test_brier"] = test[row["winner"]]["brier_score"]
        row["full_test_brier"] = test[row["full_winner"]]["brier_score"]
        row["delta_brier"] = row["test_brier"] - row["full_test_brier"]

    per_unit = aggregate(decisions, ("encoder", "domain", "budget", "method"))
    per_domain = aggregate(per_unit, ("domain", "budget", "method"))
    overall = aggregate(per_domain, ("budget", "method"))
    for row in overall:
        group = [domain for domain in per_domain
                 if (domain["budget"], domain["method"]) == (row["budget"], row["method"])]
        row["domains_over_005"] = sum(domain["delta_brier"] > 0.005 for domain in group)
        row["domain_count"] = len(group)

    write_csv(output, "per_trial.csv", decisions)
    write_csv(output, "per_encoder_domain.csv", per_unit)
    write_csv(output, "per_domain.csv", per_domain)
    write_csv(output, "overall.csv", overall)
    lines = [
        "# DomainNet fixed-pool transport replay", "",
        f"Scope: {len(keys)} encoder-domain units at projection seed 20260911. "
        "The final candidate minimizes archived logistic validation Brier within its shortlist. "
        "Test Brier gaps use the winner selected by validation over all 455 candidates. "
        "Random averages 20 archived orders. Domain means weight encoders equally, and the "
        "overall mean weights included domains equally.", "",
        "| Budget | Screening method | Test Brier | Gap to full | Domains over +0.005 |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in overall:
        lines.append("| {budget} | {method} | {test_brier:.6f} | {delta_brier:+.6f} | "
                     "{domains_over_005}/{domain_count} |".format(**row))
    lines += [
        "", "Transport is a fixed-pool adaptation. Screen-side timing and exact A-opt "
        "checks are in screen_metadata.json. The archived comparator losses do not "
        "constitute a new end-to-end timing measurement.",
    ]
    (output / "Results.md").write_text("\n".join(lines) + "\n")
    print(f"PASS: all test hashes and candidate IDs matched. Wrote {len(per_unit)} unit rows.")


if __name__ == "__main__":
    main()
