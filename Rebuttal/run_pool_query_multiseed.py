"""Replay the frozen DomainNet search on four additional projection seeds."""

import argparse
import csv
import gzip
import json
import math
import platform
from pathlib import Path
from statistics import mean

import numpy as np

import run_pool_query as frozen


RUN = frozen.REBUTTAL / "runs/domainnet-pool-query-multiseed"
SEEDS = (20260917, 20260918, 20260919, 20260920)
BASE_SEED = 20260911
TRACES = "frozen_decisions.jsonl.gz"
METRICS = ("test_brier", "full_test_brier", "delta")


def verify_freeze():
    original = frozen.verify_freeze()
    path = RUN / "freeze.sha256"
    expected = {RUN / "Protocol.md", Path(__file__).resolve(), Path(frozen.__file__).resolve()}
    checked = set()
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        source = (frozen.ROOT / name).resolve()
        assert source in expected and frozen.sha256(source.read_bytes()) == digest, name
        checked.add(source)
    assert checked == expected
    return dict(multiseed_freeze=frozen.sha256(path.read_bytes()), original_freeze=original)


def job(seed, encoder):
    folder = frozen.ARCHIVE / f"{encoder}__{seed}"
    screen = frozen.manifest(folder, "screen")
    val = frozen.manifest(folder, "validate")
    assert screen["status"] == val["status"] == "complete"
    assert screen["lineage"] == val["lineage"] and screen["context"] == val["context"]
    assert screen["context"]["projection_seed"] == seed and screen["context"]["encoder"] == encoder
    assert val["upstream"]["screen"] == screen["artifact_id"]
    return folder, screen, val


def replay_job(seed, encoder, reference):
    folder, screen, val = job(seed, encoder)
    orders = frozen.checked_json(folder / "screen", screen, "rankings.json")
    assert set(orders) == set(frozen.DOMAINS)
    if seed == BASE_SEED:
        reference[encoder] = screen["lineage"], orders
    else:
        assert screen["lineage"] == reference[encoder][0]
    rows = []
    for domain in frozen.DOMAINS:
        ranked = orders[domain]
        ids = sorted(ranked["target_a"])
        assert len(ids) == len(set(ids)) == 455
        families = {method: [ranked[method]] for method in frozen.METHODS[2:4]}
        families["random"] = [ranked[f"random:{i}"] for i in range(20)]
        assert all(sorted(order) == ids for orders_ in families.values() for order in orders_)
        base_ranks = reference[encoder][1][domain]
        assert sorted(base_ranks["target_a"]) == ids
        assert families["random"] == [base_ranks[f"random:{i}"] for i in range(20)]
        oracle = frozen.ValidationOracle(
            frozen.checked_json(folder / "validate", val, frozen.loss_file(domain)), ids)
        design, basis = frozen.pool_design(ids)
        positions = [{candidate: i for i, candidate in enumerate(ranked[method])}
                     for method in frozen.METHODS[2:4]]
        geometry = frozen.TOLERANCE * np.array([sum(r[c] for r in positions) for c in ids]) / (2 * (len(ids) - 1))
        adaptive = [frozen.trace(domain, encoder, method, 0,
                                 *frozen.adaptive_trace(ids, design, basis, offset, oracle.query))
                    for method, offset in (("geometry_pool", geometry), ("pool_only", np.zeros(len(ids))))]
        assert adaptive[0]["queries"][:15] == adaptive[1]["queries"][:15]
        rows.extend(adaptive)
        for method, orders_ in families.items():
            for repeat, order in enumerate(orders_):
                rows.append(frozen.trace(domain, encoder, method, repeat,
                                         *frozen.fixed_trace(order, oracle.query)))
    return rows


def select(header):
    path, lock = RUN / TRACES, RUN / "selection.sha256"
    if path.exists() or lock.exists():
        raise SystemExit("Frozen multiseed decisions already exist.")
    reference = {}
    base = [row for encoder in frozen.ENCODERS for row in replay_job(BASE_SEED, encoder, reference)]
    assert base == frozen.frozen_traces(header["original_freeze"])
    print(f"PASS: all {len(base)} original-seed query trajectories, revealed scores and winners reproduced exactly.")
    rows = []
    for seed in SEEDS:
        seed_rows = [dict(seed=seed, **row) for encoder in frozen.ENCODERS
                     for row in replay_job(seed, encoder, reference)]
        assert len(seed_rows) == len(base) == 288
        rows.extend(seed_rows)
        print(f"PASS {seed}: 12 units, identical candidate IDs and paired random permutations, 288 trajectories.")
    payload = "\n".join(json.dumps(row, separators=(",", ":")) for row in [header, *rows]) + "\n"
    data = gzip.compress(payload.encode(), mtime=0)
    with path.open("xb") as stream:
        stream.write(data)
    with lock.open("x") as stream:
        stream.write(f"{frozen.sha256(data)}  {path.name}\n")
    print(f"PASS: locked {len(rows)} trajectories, {len(rows) * 50} query records and "
          f"{len(rows) * len(frozen.BUDGETS)} prefix winners. Test archives were not opened.")


def load_traces(header):
    digest, name = (RUN / "selection.sha256").read_text().split()
    data = (RUN / TRACES).read_bytes()
    assert name == TRACES and frozen.sha256(data) == digest
    saved_header, *rows = [json.loads(line) for line in gzip.decompress(data).splitlines()]
    assert saved_header == header and len(rows) == 1152
    assert {r["seed"] for r in rows} == set(SEEDS)
    assert len({(r["seed"], r["encoder"], r["domain"], r["method"], r["repeat"]) for r in rows}) == len(rows)
    return rows


def read_validation(jobs):
    units = {}
    for (seed, encoder), (folder, _, val) in jobs.items():
        for domain in frozen.DOMAINS:
            scores = frozen.checked_json(folder / "validate", val, frozen.loss_file(domain))
            full = min(scores, key=lambda c: (scores[c]["brier_score"], c))
            units[seed, encoder, domain] = scores, full
    return units


def read_test(jobs, validation):
    units, archived = {}, {}
    for (seed, encoder), (folder, screen, val) in jobs.items():
        test = frozen.manifest(folder, "test-audit")
        assert test["status"] == "complete" and test["context"] == val["context"]
        assert test["lineage"] == val["lineage"]
        assert test["upstream"] == {"screen": screen["artifact_id"], "validate": val["artifact_id"]}
        for domain in frozen.DOMAINS:
            unit = seed, encoder, domain
            scores = frozen.checked_json(folder / "test-audit", test, frozen.loss_file(domain))
            assert set(scores) == set(validation[unit][0]) and len(scores) == 455
            units[unit] = scores
        baseline = frozen.checked_json(folder / "test-audit", test, "audit.json")["rows"]
        for row in baseline:
            method = row["group"].split(":")[0]
            if (row["readout"], row["metric"]) == ("logistic", "brier_score") and (
                method in frozen.METHODS[2:] and row["shortlist_size"] in (25, 50)
            ):
                assert (row["projection_seed"], row["encoder"]) == (seed, encoder)
                key = seed, encoder, row["target_domain"], row["group"], row["shortlist_size"]
                assert key not in archived
                archived[key] = row
    return units, archived


def write_csv(name, rows):
    with (RUN / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(trials):
    domains, overall = [], []
    for seed in SEEDS:
        domain_rows, summaries = frozen.summarize([r for r in trials if r["seed"] == seed])
        domains.extend(dict(seed=seed, **r) for r in domain_rows)
        overall.extend(dict(seed=seed, **r) for r in summaries)
    index = {(r["seed"], r["budget"], r["method"], r["domain"]): r for r in domains}
    for budget in frozen.BUDGETS:
        for method in frozen.METHODS:
            group = []
            for domain in frozen.DOMAINS:
                rows = [index[seed, budget, method, domain] for seed in SEEDS]
                group.append(dict(seed="mean", budget=budget, method=method, domain=domain,
                                  **{key: mean(r[key] for r in rows) for key in METRICS}))
            domains.extend(group)
            overall.append(dict(seed="mean", budget=budget, method=method,
                                **{key: mean(r[key] for r in group) for key in METRICS},
                                domains_over_005=sum(r["delta"] > frozen.TOLERANCE for r in group)))
    return domains, overall


def contrast_table(domains):
    index = {(r["seed"], r["domain"], r["budget"], r["method"]): r for r in domains}
    rows = []
    for seed in (*SEEDS, "mean"):
        for domain in frozen.DOMAINS:
            pool = index[seed, domain, 25, "pool_only"]
            geo = index[seed, domain, 25, "geometry_pool"]
            mmd = index[seed, domain, 50, "second_moment_mmd"]
            rows.append(dict(seed=seed, domain=domain, full=pool["full_test_brier"],
                             pool_25=pool["test_brier"], geometry_25=geo["test_brier"], mmd_50=mmd["test_brier"],
                             pool_vs_full=pool["delta"], geometry_vs_full=geo["delta"], mmd_vs_full=mmd["delta"],
                             pool_vs_mmd=pool["test_brier"] - mmd["test_brier"],
                             geometry_vs_mmd=geo["test_brier"] - mmd["test_brier"],
                             geometry_vs_pool=geo["test_brier"] - pool["test_brier"]))
    return rows


def report(contrast, overall):
    lines = ["# DomainNet pool-query results across additional projection seeds", "",
             "Four additional seeds, six targets, two encoders and 455 candidates per unit. "
             "The original seed is used for regression verification. Test scores follow the frozen validation choices. "
             "All differences are signed, and a negative value favors the first policy.", "",
             "## Per-domain test Brier averaged across four seeds", "",
             "| Target | Full 455 | Pool only 25 | Pool − full | MMD 50 | Pool − MMD | Geometry + pool 25 | Geometry − pool |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    mean_rows = [r for r in contrast if r["seed"] == "mean"]
    metrics = ("full", "pool_25", "pool_vs_full", "mmd_50", "pool_vs_mmd", "geometry_25", "geometry_vs_pool")
    overall_mean = dict(domain="Mean", **{k: mean(r[k] for r in mean_rows) for k in metrics})
    for row in [*mean_rows, overall_mean]:
        cells = [f"{row[k]:+.6f}" if "_vs_" in k else f"{row[k]:.6f}" for k in metrics]
        lines.append(f"| {row['domain']} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Per-seed means and domain exceedances", "",
                  "Each gap cell shows the mean signed gap and the count of six domains exceeding +0.005.", "",
                  "| Seed | Pool 25 − MMD 50 | Pool 25 − full | Geometry 25 − MMD 50 | Geometry 25 − full | Geometry 25 − pool 25 |",
                  "|---|---:|---:|---:|---:|---:|"])
    keys = ("pool_vs_mmd", "pool_vs_full", "geometry_vs_mmd", "geometry_vs_full", "geometry_vs_pool")
    for seed in (*SEEDS, "mean"):
        rows = [r for r in contrast if r["seed"] == seed]
        cells = [f"{mean(r[k] for r in rows):+.6f} ({sum(r[k] > frozen.TOLERANCE for r in rows)}/6)" for k in keys]
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")
    lines.extend(["", "| Contrast | Seed–domain gaps > +0.005 | Domains with any seed > +0.005 | Worst seed–domain gap |",
                  "|---|---:|---:|---:|"])
    seed_domains = [r for r in contrast if r["seed"] != "mean"]
    for key in keys:
        tail = [r for r in seed_domains if r[key] > frozen.TOLERANCE]
        worst = max(seed_domains, key=lambda r: r[key])
        lines.append(f"| {key} | {len(tail)}/24 | {len({r['domain'] for r in tail})}/6 | "
                     f"{worst[key]:+.6f} ({worst['domain']}, {worst['seed']}) |")
    lines.extend(["", "## All policies averaged across four seeds", "",
                  "| Budget | Method | Test Brier | Gap to full | Domains with gap > +0.005 |", "|---:|---|---:|---:|---:|"])
    for row in (r for r in overall if r["seed"] == "mean"):
        lines.append(f"| {row['budget']} | {frozen.LABELS[row['method']]} | {row['test_brier']:.6f} | "
                     f"{row['delta']:+.6f} | {row['domains_over_005']}/6 |")
    lines.extend(["", "The [primary contrasts](primary_contrast.csv) include test Brier and signed gaps for every seed and domain. "
                  "The [per-domain CSV](per_domain.csv) includes all budgets and methods. The seeds use the same underlying data "
                  "and candidate pools. Query counts describe the archived-fit replay."])
    (RUN / "Results.md").write_text("\n".join(lines) + "\n")


def audit(header):
    traces = load_traces(header)
    jobs = {(seed, encoder): job(seed, encoder) for seed in SEEDS for encoder in frozen.ENCODERS}
    validation = read_validation(jobs)
    for row in traces:
        scores, _ = validation[row["seed"], row["encoder"], row["domain"]]
        queries, observed = row["queries"], row["validation_brier"]
        assert len(set(queries)) == len(queries) == len(observed) == 50
        assert observed == [scores[c]["brier_score"] for c in queries]
        assert row["winners"] == frozen.choose_winners(queries, observed)
    print("PASS: all frozen query scores and prefix winners verified before opening the test archives.")
    test, archived = read_test(jobs, validation)
    trials, matched = [], 0
    for row in traces:
        unit = row["seed"], row["encoder"], row["domain"]
        full = validation[unit][1]
        full_score = test[unit][full]["brier_score"]
        for budget in frozen.BUDGETS:
            winner = row["winners"][str(budget)]
            score = test[unit][winner]["brier_score"]
            method = row["method"]
            if budget in (25, 50) and method in frozen.METHODS[2:]:
                group = f"random:{row['repeat']}" if method == "random" else method
                check = archived[(*unit, group, budget)]
                assert winner == check["combination"] and score == check["test_loss"]
                matched += 1
            trials.append(dict(seed=unit[0], encoder=unit[1], domain=unit[2], method=method, repeat=row["repeat"],
                               budget=budget, winner=winner, full_winner=full, test_brier=score,
                               full_test_brier=full_score, delta=score - full_score))
    assert matched == len(archived) == 2112
    domains, overall = aggregate(trials)
    budget_audit = frozen.REBUTTAL / "runs/replay-001/recomputed/budget/curves.csv"
    with budget_audit.open() as source:
        reference, = (r for r in csv.DictReader(source)
                      if (r["readout"], r["metric"], r["method"], r["shortlist_size"])
                      == ("logistic", "brier_score", "exhaustive", "455"))
    mean_full = next(r["full_test_brier"] for r in overall if r["seed"] == "mean")
    assert int(reference["repeated_task_count"]) == len(test) == 48
    assert math.isclose(mean_full, float(reference["test_loss"]), rel_tol=0, abs_tol=1e-12)
    print(f"PASS: all {matched} archived fixed-baseline winners and scores match. "
          f"Mean full-validation Brier {mean_full:.12f} matches the four-seed budget audit.")
    contrast = contrast_table(domains)
    for name, rows in (("per_trial", trials), ("per_domain", domains), ("overall", overall), ("primary_contrast", contrast)):
        write_csv(f"{name}.csv", rows)
    report(contrast, overall)
    print(f"PASS: wrote {len(trials)} trial rows, {len(domains)} domain rows, {len(overall)} summaries and {len(contrast)} contrasts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "audit"))
    args = parser.parse_args()
    print(f"Phase {args.phase}. Python {platform.python_version()}, NumPy {np.__version__}.")
    header = verify_freeze()
    print(f"PASS: original and multiseed protocol/code hashes verified. {header}")
    (select if args.phase == "select" else audit)(header)
