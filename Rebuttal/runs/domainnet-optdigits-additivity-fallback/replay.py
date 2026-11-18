"""Freeze, then replay 25-fit additivity checks from validation archives only."""

import argparse
import hashlib
import importlib.util
import json
import math
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np


THRESHOLD = 0.005
INITIAL_FITS = 15
PROBES = 3
BUDGET = 25
CANDIDATES = list(combinations(range(INITIAL_FITS), 3))
IDS = set(range(len(CANDIDATES)))
BASELINES = ("sequential25", "mmd25", "mmd50", "full455")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def policy_source(repo):
    return repo / "fresh_20261117" / "policies.py"


def archive_dir(repo):
    return repo / "runs" / "fresh-20261117"


def get_units(repo):
    domains = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
    seeds = range(20261117, 20261121)
    names = [
        f"domainnet_{domain}_{partition}_d128_{seed}"
        for domain in domains for partition in ("hash", "pc1") for seed in seeds
    ] + [
        f"optdigits_{partition}_{seed}"
        for partition in ("hash", "pc1") for seed in seeds
    ]
    return [archive_dir(repo) / name for name in names]


def frozen_inputs(repo, run_dir):
    units = {}
    for unit in get_units(repo):
        dec_hash = sha(unit / "decisions.json")
        lock = unit / "selection_lock.json"
        assert dec_hash == load(lock)["decisions"], unit.name
        units[unit.name] = {"decisions": dec_hash, "selection_lock": sha(lock)}
    return {
        "protocol_sha256": sha(run_dir / "Protocol.md"),
        "replay_sha256": sha(Path(__file__)),
        "original_policy_sha256": sha(policy_source(repo)),
        "original_units": units,
        "initial_fits": INITIAL_FITS,
        "diagnostic_fits": PROBES,
        "budget": BUDGET,
        "threshold": THRESHOLD,
    }


def original_module(repo):
    spec = importlib.util.spec_from_file_location("original_policies", policy_source(repo))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def leverage(design, queried):
    x = design[queried]
    chol = np.linalg.cholesky(x.T @ x)
    return np.linalg.norm(np.linalg.solve(chol, design.T), axis=0)


def rank_min(values, queried):
    return min(IDS.difference(queried), key=lambda i: (values[i], i))


def diagnostic(design, initial, reveal):
    """Choose probe IDs without outcomes, then evaluate the held-out predictions."""
    observed = [reveal(i) for i in initial]
    x = design[initial]
    beta = np.linalg.lstsq(x, observed, rcond=None)[0]
    initial_leverage = leverage(design, initial)
    probes = []
    for _ in range(PROBES):
        used = initial + probes
        pick = rank_min(-np.round(leverage(design, used) ** 2, 12), used)
        probes.append(pick)
    actual = [reveal(i) for i in probes]
    pred = design[probes] @ beta
    residuals = np.abs(np.asarray(actual) - pred) / np.sqrt(1 + initial_leverage[probes] ** 2)
    score = float(np.median(residuals))
    return SimpleNamespace(
        prefix=initial + probes, observed=observed + actual, probes=probes,
        predicted=pred.tolist(), residuals=residuals.tolist(), score=score,
    )


def complete(design, prefix, observed, mmd_order, reveal, switch):
    queries, values = list(prefix), list(observed)
    if switch:
        remaining = [i for i in mmd_order if i not in queries][:BUDGET - len(queries)]
        queries.extend(remaining)
        values.extend(reveal(i) for i in remaining)
    else:
        while len(queries) < BUDGET:
            beta = np.linalg.lstsq(design[queries], values, rcond=None)[0]
            scores = np.round(design @ beta - 0.001 * leverage(design, queries), 12)
            winner = rank_min(scores, queries)
            queries.append(winner)
            values.append(reveal(winner))
    assert len(queries) == len(set(queries)) == BUDGET
    winner = min(zip(values, queries))[1]
    return {"queries": queries, "observed": values, "winner": winner}


def archived_decisions(unit):
    doc = load(unit / "decisions.json")
    records = {
        row["method"]: row for row in doc["decisions"] if row["method"] in BASELINES
    }
    assert set(records) == set(BASELINES)
    full = records["full455"]
    assert set(full["queries"]) == IDS and len(full["observed"]) == len(IDS)
    losses = dict(zip(full["queries"], full["observed"]))
    assert all(math.isfinite(value) for value in losses.values())
    order = doc["rankings"]["mmd"]
    assert len(order) == len(IDS) and set(order) == IDS
    return records, losses, order


def replay_unit(unit, design, initial, original):
    records, validation, order = archived_decisions(unit)

    def verify_trace(result, revealed):
        assert result["queries"] == revealed
        assert result["observed"] == [validation[i] for i in revealed]
        assert result["winner"] == min(revealed, key=lambda i: (validation[i], i))
        return {key: result[key] for key in ("queries", "observed", "winner")}

    def reveal(i):
        assert i not in visible
        visible.append(i)
        return float(validation[i])

    policies = {}
    for name, kind, budget in [
        ("sequential25", "sequential", 25), ("mmd25", "mmd", 25),
        ("mmd50", "mmd", 50), ("full455", "full", 455),
    ]:
        visible = []
        rank = list(range(len(IDS))) if kind == "full" else order
        result = original.run_policy(kind, design, initial, reveal, rank, budget)
        checked = verify_trace(result, visible)
        assert all(checked[k] == records[name][k] for k in checked), (unit.name, name)
        policies[name] = checked

    visible = []
    check = diagnostic(design, initial, reveal)
    assert visible == check.prefix
    triggered = check.score > THRESHOLD
    for name, switch in (("hybrid", triggered), ("continue25", False), ("switch25", True)):
        visible = list(check.prefix)
        result = complete(design, check.prefix, check.observed, order, reveal, switch)
        policies[name] = verify_trace(result, visible)
    assert policies["hybrid"] == policies["switch25" if triggered else "continue25"]

    split = unit.name.split("_")
    dataset = split[0]
    domain = split[1] if dataset == "domainnet" else "optdigits"
    partition = split[2] if dataset == "domainnet" else split[1]
    return {
        "unit": unit.name, "dataset": dataset, "domain": domain,
        "partition": partition, "seed": int(split[-1]), "probe_ids": check.probes,
        "probe_predictions": check.predicted, "probe_residuals": check.residuals,
        "diagnostic_score": check.score, "triggered": triggered, "policies": policies,
    }


def self_test(design, initial):
    beta = np.linspace(0.02, 0.1, INITIAL_FITS)
    losses = design @ beta
    check = diagnostic(design, initial, losses.__getitem__)
    assert check.score < 1e-12
    changed = losses.copy()
    changed[check.probes] += 0.1
    bad = diagnostic(design, initial, changed.__getitem__)
    assert bad.probes == check.probes and bad.score > THRESHOLD
    rank = list(reversed(range(len(IDS))))
    returned = complete(design, bad.prefix, bad.observed, rank, changed.__getitem__, True)
    assert returned["queries"][18:] == [i for i in rank if i not in bad.prefix][:7]
    assert returned["winner"] == min(returned["queries"], key=lambda i: (changed[i], i))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["freeze", "select"])
    parser.add_argument("--repo", type=Path, required=True, help="Rebuttal directory of archive clone")
    run_dir = Path(__file__).resolve().parent
    args = parser.parse_args()
    snapshot = frozen_inputs(args.repo, run_dir)
    if args.phase == "freeze":
        save_new(run_dir / "freeze.json", snapshot)
        print("FROZEN", sha(run_dir / "freeze.json"), "56 source validation archives")
        return
    assert snapshot == load(run_dir / "freeze.json"), "Frozen input changed"
    original = original_module(args.repo)
    design, initial = original.design_basis(CANDIDATES, INITIAL_FITS)
    self_test(design, initial)
    print("PASS synthetic exact-additivity and force-switch checks")
    selected = [replay_unit(unit, design, initial, original) for unit in get_units(args.repo)]
    save_new(run_dir / "selection.json", selected)
    save_new(run_dir / "selection.lock.json", {
        "selection_sha256": sha(run_dir / "selection.json"),
        "freeze_sha256": sha(run_dir / "freeze.json"),
    })
    print("LOCKED", sha(run_dir / "selection.json"), f"{len(selected)} units; "
          f"{len(BASELINES)*len(selected)} baseline traces aligned")
    print("Test files unopened during freeze and selection")


if __name__ == "__main__":
    main()
