"""Replay the base-seed DomainNet logistic ablation after checking frozen rankings."""
import csv
import json
import hashlib
from pathlib import Path
from statistics import mean

runs = Path(__file__).resolve().parent / "runs"
base = runs / "replay-001/package/results/target_conditioned"
out = runs / "domainnet-logistic-ablation"
out.mkdir(exist_ok=True)
budgets = (10, 25, 50, 100, 227)
methods = ("target_a", "target_energy", "second_moment_mmd")
domains = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
units = []
checks = []
for encoder in ("resnet50", "dinov2_b14"):
    records = base / "revision_20260917/revision_results/e2b_readout_projection_v1" / f"{encoder}__20260911"
    screen, validate, audit = [
        json.loads((records / stage / "manifest.json").read_text())
        for stage in ("screen", "validate", "test-audit")
    ]
    assert screen["lineage"] == validate["lineage"] == audit["lineage"]
    assert validate["upstream"]["screen"] == audit["upstream"]["screen"] == screen["artifact_id"]
    assert audit["upstream"]["validate"] == validate["artifact_id"]
    for manifest in (screen, validate, audit):
        for name, digest in manifest["files"].items():
            if name.endswith(".npz"):
                continue
            assert hashlib.sha256((records / manifest["stage"] / name).read_bytes()).hexdigest() == digest
    rankings = json.loads((records / "screen/rankings.json").read_text())
    with (base / "e2b_fixed_cost_shortlist/domainnet_v1/screen" / encoder / "rankings.csv").open() as stream:
        reference = list(csv.DictReader(stream))
    for domain in domains:
        orders = rankings[domain]
        candidates = set(orders["target_a"])
        assert len(candidates) == 455
        for method in methods:
            rows = sorted((r for r in reference if r["target_domain"] == domain and r["method"] == method),
                          key=lambda r: int(r["rank"]))
            old = [r["combination"] for r in rows]
            new = orders[method]
            assert len(new) == len(old) == 455 and set(old) == set(new) == candidates
            overlap = [len(set(old[:k]) & set(new[:k])) for k in budgets]
            checks.append(dict(encoder=encoder, domain=domain, method=method,
                               exact_order=old == new, overlaps=overlap))
        units.append((encoder, domain, records, orders, candidates))
(out / "ranking_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
differences = sum(k != n for row in checks for k, n in zip(budgets, row["overlaps"]))
print(f"Cross-archive comparison: {differences} shortlist differences. "
      "Using the checked screen/validate/test-audit lineage throughout.")
print(f"PASS: {len(checks)} ranking comparisons; exact orders {sum(r['exact_order'] for r in checks)}", flush=True)

rows = []
metrics = ("test_brier", "omission", "selection")
for encoder, domain, records, orders, candidates in units:
    val, test = [json.loads((records / stage / f"{domain}__logistic_losses.json").read_text())
                 for stage in ("validate", "test-audit")]
    assert set(val) == set(test) == candidates
    oracle = min(row["brier_score"] for row in test.values())

    def evaluate(prefix):
        pick = min(prefix, key=lambda c: (val[c]["brier_score"], c))
        best = min(test[c]["brier_score"] for c in prefix)
        loss = test[pick]["brier_score"]
        return loss, best - oracle, loss - best

    random_orders = [orders[f"random:{repeat}"] for repeat in range(20)]
    assert all(len(order) == 455 and set(order) == candidates for order in random_orders)
    for k in budgets:
        values = {method: evaluate(orders[method][:k]) for method in methods}
        values["random_mean"] = tuple(map(mean, zip(*(evaluate(order[:k]) for order in random_orders))))
        for method, value in values.items():
            rows.append(dict(encoder=encoder, domain=domain, budget=k, method=method,
                             **dict(zip(metrics, value))))
print(f"PASS: {len(units)} encoder-domain units with 455 paired validation/test candidates each")

def write(name, table):
    with (out / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

write("per_encoder_domain.csv", rows)
def aggregate(domain):
    return [dict(domain=domain, budget=k, method=method,
                 **{metric: mean(r[metric] for r in rows if r["budget"] == k and r["method"] == method
                                 and (domain == "overall" or r["domain"] == domain)) for metric in metrics})
            for k in budgets for method in (*methods, "random_mean")]
write("per_domain.csv", [row for domain in domains for row in aggregate(domain)])
write("overall.csv", aggregate("overall"))
