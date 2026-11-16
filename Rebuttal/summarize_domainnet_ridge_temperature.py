"""Summarize the verified two-phase DomainNet ridge calibration replay."""
import csv
import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent / "runs/domainnet-ridge-temperature"
domains = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
encoders = ("resnet50", "dinov2_b14")
metrics = ("brier", "accuracy", "delta_brier", "delta_accuracy", "temperature", "mean_max_probability")
units, gates = [], []
for domain in domains:
    for encoder in encoders:
        folder = root / encoder / domain
        gate = json.loads((folder / "t1_gate.json").read_text())
        assert max(gate.values()) <= 1e-4
        gates.append(dict(domain=domain, encoder=encoder, **gate))
        with (folder / "results.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 444
        full = {r["arm"]: r for r in rows if r["method"] == "full"}
        by = {(r["budget"], r["method"], r["arm"]): r for r in rows}
        for row in rows:
            if row["arm"] == "original":
                calibrated = by[row["budget"], row["method"], "original_calibrated"]
                assert (row["candidate"], row["accuracy"]) == (calibrated["candidate"], calibrated["accuracy"])
        for budget in ("50", "227", "455"):
            for arm, baseline in full.items():
                for method in ("target_a", "second_moment_mmd", "random", "full"):
                    group = [r for r in rows if r["budget"] == budget and r["arm"] == arm
                             and r["method"].split(":")[0] == method]
                    if not group:
                        continue
                    assert len(group) == (20 if method == "random" else 1)
                    brier = mean(float(r["brier"]) for r in group)
                    accuracy = mean(float(r["accuracy"]) for r in group)
                    units.append(dict(domain=domain, encoder=encoder, budget=budget, arm=arm,
                                      method=method, brier=brier, accuracy=accuracy,
                                      delta_brier=brier-float(baseline["brier"]),
                                      delta_accuracy=accuracy-float(baseline["accuracy"]),
                                      temperature=mean(float(r["temperature"]) for r in group),
                                      mean_max_probability=mean(float(r["mean_max_probability"]) for r in group)))

def write(name, rows):
    with (root / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def aggregate(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    return [dict(zip(keys, key), **{m: mean(r[m] for r in group) for m in metrics})
            for key, group in groups.items()]

per_domain = aggregate(units, ("domain", "budget", "arm", "method"))
overall = aggregate(per_domain, ("budget", "arm", "method"))
write("gates.csv", gates)
write("per_encoder_domain.csv", units)
write("per_domain.csv", per_domain)
write("overall.csv", overall)
print(f"PASS: {len(gates)} gates, {len(units)} encoder-domain rows, {len(per_domain)} domain rows.")

def markdown_table(rows, include_domain=False):
    headings = (["Domain"] if include_domain else []) + [
        "Budget", "Method", "Arm", "Brier", "Accuracy", "Brier gap", "Accuracy gap (pp)"]
    lines = ["| " + " | ".join(headings) + " |", "|" + "|".join(["---"] * len(headings)) + "|"]
    for row in rows:
        values = ([row["domain"]] if include_domain else []) + [
            row["budget"], row["method"], row["arm"], f'{row["brier"]:.6f}',
            f'{row["accuracy"]:.2%}', f'{row["delta_brier"]:+.6f}',
            f'{100 * row["delta_accuracy"]:+.2f}']
        lines.append("| " + " | ".join(values) + " |")
    return lines

largest = {stage: max(row[stage] for row in gates) for stage in ("validation", "test")}
lines = [
    "# DomainNet ridge temperature calibration", "",
    f"All 12 T=1 gates passed. Maximum absolute Brier errors were {largest['validation']:.3g} "
    f"on validation and {largest['test']:.3g} on test.", "",
    "Brier is lower-is-better. Accuracy is top-1. Signed gaps are relative to the "
    "455-fit validation winner under the same arm. Random averages 20 orders per unit. "
    "Domain averages weight the two encoders equally, and the overall average weights all six domains equally.", "",
    "Original winners use full validation for temperature fitting. Reselection compares both arms "
    "on the same held-out selection half of validation. Uniform Brier is 0.99.", "",
    "## Overall results", "", *markdown_table(overall), "",
    "## Per-domain calibrated results", "",
    *markdown_table([r for r in per_domain if r["arm"].endswith("_calibrated")], True), "",
    "All arms are available in per_domain.csv and per_encoder_domain.csv.",
]
(root / "Results.md").write_text("\n".join(lines) + "\n")
