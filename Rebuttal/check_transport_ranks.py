"""Check rebuilt baseline ranks against both frozen DomainNet references."""
import csv
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
archive = Path(__file__).resolve().parent / "runs/replay-001/package/results/target_conditioned"
checks = []
for encoder in ("resnet50", "dinov2_b14"):
    primary = archive / f"e2b_fixed_cost_shortlist/domainnet_v1/screen/{encoder}/rankings.csv"
    with primary.open() as stream:
        rows = list(csv.DictReader(stream))
    paired = json.loads((archive / "revision_20260917/revision_results/e2b_readout_projection_v1"
                         / f"{encoder}__20260911/screen/rankings.json").read_text())
    for domain, methods in sorted(paired.items()):
        screen = json.loads((root / "units" / encoder / domain / "screen.json").read_text())
        assert (screen["encoder"], screen["domain"]) == (encoder, domain)
        for method in ("target_a", "second_moment_mmd"):
            new = screen["geometry_baselines"][method]["ranking"]
            old_rows = sorted((r for r in rows if (r["target_domain"], r["method"]) == (domain, method)),
                              key=lambda row: int(row["rank"]))
            old = [r["combination"] for r in old_rows]
            assert len(new) == len(old) == len(set(new)) == 455
            for source, reference in (("primary", old), ("paired", methods[method])):
                assert set(reference) == set(new)
                position = {key: i for i, key in enumerate(reference)}
                rho = 1 - 6 * sum((i - position[key]) ** 2 for i, key in enumerate(new)) / (455 * (455 ** 2 - 1))
                overlap = {k: len(set(new[:k]) & set(reference[:k])) for k in (50, 227)}
                checks.append(dict(encoder=encoder, domain=domain, method=method, reference=source,
                                   spearman=rho, top50_overlap=overlap[50], top227_overlap=overlap[227]))
with (root / "ranking_checks.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(checks[0]))
    writer.writeheader()
    writer.writerows(checks)
print(f"PASS: candidate IDs match in all {len(checks)} comparisons.")
for source in ("primary", "paired"):
    group = [r for r in checks if r["reference"] == source]
    differences = sum(r["top50_overlap"] != 50 for r in group) + sum(r["top227_overlap"] != 227 for r in group)
    print(f"{source}: min rho={min(r['spearman'] for r in group):.9f}, "
          f"{differences}/{2*len(group)} shortlist boundary differences")
    for row in group:
        if row["top50_overlap"] != 50 or row["top227_overlap"] != 227:
            print(row)
