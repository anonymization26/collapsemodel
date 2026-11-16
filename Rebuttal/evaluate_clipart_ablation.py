"""Check the rebuilt ranking before opening the frozen logistic losses."""
import csv
import json
from pathlib import Path
import statistics
import sys

from scipy.stats import spearmanr

run = Path(__file__).resolve().parent / "runs"
pilot = run / "clipart-resnet50"
base = run / "replay-001/package/results/target_conditioned"
reference = base / "e2b_fixed_cost_shortlist/domainnet_v1/screen/resnet50/rankings.csv"
with reference.open() as stream:
    rows = [row for row in csv.DictReader(stream)
            if row["target_domain"] == "clipart" and row["method"] == "target_a"]
line = next(line for line in Path(sys.argv[1]).read_text().splitlines()
            if line.startswith("SCORES_JSON="))
scores = json.loads(line.split("=", 1)[1])
old = [row["combination"] for row in sorted(rows, key=lambda row: int(row["rank"]))]
rankings = {method: sorted(values, key=lambda key: (
    (-1 if method == "target_energy" else 1) * values[key], key))
    for method, values in scores.items()}
new = rankings["target_a"]
assert len(old) == len(new) == 455 and set(old) == set(new)
position = {key: i for i, key in enumerate(new)}
rho = float(spearmanr(range(len(old)), [position[key] for key in old]).statistic)
budgets = [10, 25, 50, 100, 227]
overlap = {size: len(set(old[:size]) & set(new[:size])) for size in budgets}
alignment = dict(spearman=rho, overlap=overlap, exact_order=(old == new),
                 max_abs_score_error=max(abs(float(row["score"]) - scores["target_a"][row["combination"]])
                                         for row in rows))
(pilot / "alignment.json").write_text(json.dumps(alignment, indent=2) + "\n")
if rho < 0.999 or any(overlap[size] != size for size in budgets):
    sys.exit("A-opt ranking gate failed. See alignment.json.")
records = base / "revision_20260917/revision_results/e2b_readout_projection_v1/resnet50__20260911"
archived = json.loads((records / "screen/rankings.json").read_text())["clipart"]
assert all(set(archived[method]) == set(old) for method in rankings)
assert all(set(archived["target_a"][:size]) == set(new[:size]) for size in budgets)
validation = json.loads((records / "validate/clipart__logistic_losses.json").read_text())
test = json.loads((records / "test-audit/clipart__logistic_losses.json").read_text())
assert set(validation) == set(test) == set(old)
oracle = min(value["brier_score"] for value in test.values())

def evaluate(prefix):
    pick = min(prefix, key=lambda key: (validation[key]["brier_score"], key))
    best = min(test[key]["brier_score"] for key in prefix)
    loss = test[pick]["brier_score"]
    return dict(test_brier=loss, omission=best - oracle, selection=loss - best)

result = {}
for size in budgets:
    summary = {method: evaluate(order[:size]) for method, order in rankings.items()}
    trials = [evaluate(order[:size]) for method, order in archived.items() if method.startswith("random:")]
    summary["random_mean"] = {key: statistics.mean(row[key] for row in trials) for key in trials[0]}
    result[size] = summary
(pilot / "logistic_ablation.json").write_text(json.dumps(result, indent=2) + "\n")
