"""Replay frozen DomainNet rankings using validation-only budget calibration."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

runs = Path(__file__).resolve().parent / "runs"
out = runs / "domainnet-query-efficiency"
archive = runs / ("replay-001/package/results/target_conditioned/"
                  "revision_20260917/revision_results/e2b_readout_projection_v1")
domains = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
encoders = ("resnet50", "dinov2_b14")
methods = ("target_a", "second_moment_mmd", "random")
thresholds = (0, .00025, .0005, .001, .002)

def checked_json(folder, manifest, name):
    data = (folder / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest["files"][name]
    return json.loads(data)

def pick(order, loss):
    return min(order, key=lambda key: (loss[key]["brier_score"], key))

def adaptive(order, loss, threshold):
    incumbent = pick(order[:10], loss)
    previous = loss[incumbent]["brier_score"]
    start = 10
    for size in (25, 50, 100, 227):
        incumbent = pick([incumbent, *order[start:size]], loss)
        current = loss[incumbent]["brier_score"]
        if previous - current <= threshold or size == 227:
            return size, incumbent
        previous, start = current, size

tasks = {}
for encoder in encoders:
    folder = archive / f"{encoder}__20260911"
    screen, validate, test = [
        json.loads((folder / stage / "manifest.json").read_text())
        for stage in ("screen", "validate", "test-audit")
    ]
    assert screen["lineage"] == validate["lineage"] == test["lineage"]
    assert validate["upstream"]["screen"] == test["upstream"]["screen"] == screen["artifact_id"]
    assert test["upstream"]["validate"] == validate["artifact_id"]
    orders = checked_json(folder / "screen", screen, "rankings.json")
    for domain in domains:
        name = f"{domain}__logistic_losses.json"
        loss = checked_json(folder / "validate", validate, name)
        ranked = {method: [orders[domain][method]] for method in methods[:2]}
        ranked["random"] = [orders[domain][f"random:{i}"] for i in range(20)]
        assert len(loss) == 455
        assert all(len(order) == 455 and set(order) == set(loss)
                   for family in ranked.values() for order in family)
        tasks[domain, encoder] = loss, ranked, pick(loss, loss), folder, test

print("PASS: verified ranking and validation hashes, lineage, and 455 candidate IDs in all 12 units.")
decisions = []
folds = {}
for held in domains:
    calibration = {}
    for threshold in thresholds:
        counts, gaps = [], {}
        for method in methods:
            unit_gaps = []
            for (domain, _), (loss, ranked, full, _, _) in tasks.items():
                if domain == held:
                    continue
                trials = [adaptive(order, loss, threshold) for order in ranked[method]]
                counts.append(mean(size for size, _ in trials))
                unit_gaps.append(mean(loss[key]["brier_score"] - loss[full]["brier_score"]
                                      for _, key in trials))
            gaps[method] = mean(unit_gaps)
        calibration[threshold] = dict(queries=mean(counts), gaps=gaps)
    def priority(threshold):
        row = calibration[threshold]
        worst = max(row["gaps"].values())
        return (0, row["queries"], threshold) if worst <= .005 else (1, worst, row["queries"], threshold)

    threshold = min(thresholds, key=priority)
    folds[held] = dict(threshold=threshold, calibration=calibration)
    print(f"{held}: frozen threshold {threshold:g}")
    for encoder in encoders:
        loss, ranked, full, _, _ = tasks[held, encoder]
        for method, orders in ranked.items():
            for repeat, order in enumerate(orders):
                for policy, (size, winner) in (
                    ("adaptive", adaptive(order, loss, threshold)),
                    ("fixed50", (50, pick(order[:50], loss))),
                ):
                    decisions.append(dict(domain=held, encoder=encoder, method=method,
                                          repeat=repeat, policy=policy, queries=size,
                                          winner=winner, full_winner=full))

(out / "frozen_decisions.json").write_text(json.dumps(dict(folds=folds, decisions=decisions), indent=2) + "\n")
print("PASS: all decisions saved before opening test losses.")
test_losses = {}
for (domain, encoder), (loss, _, _, folder, manifest) in tasks.items():
    test = checked_json(folder / "test-audit", manifest, f"{domain}__logistic_losses.json")
    assert set(test) == set(loss)
    test_losses[domain, encoder] = test
for row in decisions:
    test = test_losses[row["domain"], row["encoder"]]
    row["test_brier"] = test[row["winner"]]["brier_score"]
    row["full_test_brier"] = test[row["full_winner"]]["brier_score"]
    row["delta"] = row["test_brier"] - row["full_test_brier"]

def write(name, rows):
    with (out / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

per_domain, overall = [], []
for method in methods:
    for policy in ("adaptive", "fixed50"):
        group = []
        for domain in domains:
            rows = [r for r in decisions if (r["domain"], r["method"], r["policy"]) == (domain, method, policy)]
            stats = {k: mean(mean(r[k] for r in rows if r["encoder"] == encoder)
                             for encoder in encoders)
                     for k in ("queries", "test_brier", "full_test_brier", "delta")}
            group.append(dict(domain=domain, method=method, policy=policy, **stats))
        per_domain.extend(group)
        overall.append(dict(method=method, policy=policy,
                            **{k: mean(r[k] for r in group) for k in stats},
                            domains_over_005=sum(r["delta"] > .005 for r in group)))
write("per_trial.csv", decisions)
write("per_domain.csv", per_domain)
write("overall.csv", overall)
print("PASS: test hashes and candidate IDs verified. Wrote 36 domain rows and 6 summaries.")
