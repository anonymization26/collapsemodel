"""Compare a single second batch with frozen sequential DomainNet queries."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np

import run_pool_query as base
import run_pool_query_multiseed as multi


RUN = base.REBUTTAL / "runs/domainnet-pool-query-batch"
DECISIONS = RUN / "frozen_decisions.json"
LOCK = RUN / "selection.sha256"
BUDGETS = (25, 50)


def digest(path):
    return base.sha256(path.read_bytes())


def fixed_batch(ids, design, basis, reveal):
    queried = [ids[i] for i in basis]
    values = [reveal(candidate) for candidate in queried]
    x = design[basis]
    beta = np.linalg.lstsq(x, values, rcond=None)[0]
    lower = np.linalg.cholesky(x.T @ x)
    leverage = np.linalg.norm(np.linalg.solve(lower, design.T), axis=0)
    scores = np.round(design @ beta - base.EXPLORATION * leverage, 12)
    order = sorted(set(range(len(ids))) - set(basis), key=lambda i: (scores[i], ids[i]))
    second_batch = [ids[i] for i in order[:BUDGETS[-1] - len(basis)]]
    return queried + second_batch, values + [reveal(candidate) for candidate in second_batch]


def select(header, archived):
    rows = []
    for seed in multi.SEEDS:
        for encoder in base.ENCODERS:
            folder, screen, val = multi.job(seed, encoder)
            orders = base.checked_json(folder / "screen", screen, "rankings.json")
            for domain in base.DOMAINS:
                ids = sorted(orders[domain]["target_a"])
                design, basis = base.pool_design(ids)
                losses = base.checked_json(folder / "validate", val, base.loss_file(domain))
                oracle = base.ValidationOracle(losses, ids)
                queries, scores = fixed_batch(ids, design, basis, oracle.query)
                prior = archived[seed, encoder, domain, "pool_only"]
                assert queries[:15] == prior["queries"][:15]
                assert scores[:15] == prior["validation_brier"][:15]
                assert len(set(queries)) == len(queries) == len(scores) == 50
                rows.append(dict(seed=seed, encoder=encoder, domain=domain, queries=queries,
                                 validation_brier=scores, winners=base.choose_winners(queries, scores)))
    with DECISIONS.open("x") as file:
        json.dump({"freeze": header, "rows": rows}, file)
    with LOCK.open("x") as file:
        file.write(f"{digest(DECISIONS)}  {DECISIONS.name}\n")
    print(f"Locked {len(rows)} trajectories before test audit.")


def audit(header, archived):
    assert LOCK.read_text().split() == [digest(DECISIONS), DECISIONS.name]
    frozen = json.loads(DECISIONS.read_text())
    assert frozen["freeze"] == header and len(frozen["rows"]) == 48
    jobs = {(s, e): multi.job(s, e) for s in multi.SEEDS for e in base.ENCODERS}
    validation = multi.read_validation(jobs)
    for row in frozen["rows"]:
        unit = row["seed"], row["encoder"], row["domain"]
        scores, _ = validation[unit]
        assert row["validation_brier"] == [scores[c]["brier_score"] for c in row["queries"]]
        assert row["winners"] == base.choose_winners(row["queries"], row["validation_brier"])
    tests = {}
    for (seed, encoder), (folder, screen, val) in jobs.items():
        test = base.manifest(folder, "test-audit")
        assert test["status"] == "complete" and test["context"] == val["context"]
        assert test["lineage"] == val["lineage"]
        assert test["upstream"] == {"screen": screen["artifact_id"], "validate": val["artifact_id"]}
        for domain in base.DOMAINS:
            unit = seed, encoder, domain
            scores = base.checked_json(folder / "test-audit", test, base.loss_file(domain))
            assert set(scores) == set(validation[unit][0]) and len(scores) == 455
            tests[unit] = scores
    units = []
    for row in frozen["rows"]:
        unit = row["seed"], row["encoder"], row["domain"]
        sequential = archived[*unit, "pool_only"]["winners"]
        mmd = archived[*unit, "second_moment_mmd"]["winners"]["50"]
        full = validation[unit][1]
        for budget in BUDGETS:
            batch = row["winners"][str(budget)]
            seq = sequential[str(budget)]
            a, b, c, d = (tests[unit][i]["brier_score"] for i in (batch, seq, mmd, full))
            units.append(dict(seed=unit[0], encoder=unit[1], domain=unit[2], budget=budget,
                              batch=a, sequential=b, mmd50=c, full455=d,
                              batch_minus_sequential=a-b, batch_minus_mmd50=a-c,
                              batch_minus_full=a-d, sequential_minus_mmd50=b-c,
                              sequential_minus_full=b-d, same_winner=batch == seq))
    domain_rows = []
    for budget in BUDGETS:
        for seed in multi.SEEDS:
            for domain in base.DOMAINS:
                pair = [r for r in units if (r["budget"], r["seed"], r["domain"]) == (budget, seed, domain)]
                assert len(pair) == 2
                metrics = tuple(units[0])[4:-1]
                domain_rows.append(dict(budget=budget, seed=seed, domain=domain,
                                        **{k: mean(r[k] for r in pair) for k in metrics}))
    for path, rows in (("per_unit.csv", units), ("per_seed_domain.csv", domain_rows)):
        with (RUN / path).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    for budget in BUDGETS:
        rows = [r for r in domain_rows if r["budget"] == budget]
        print(f"Budget {budget}, test Brier " + ", ".join(
            f"{k}={mean(r[k] for r in rows):.9f}" for k in ("batch", "sequential", "mmd50", "full455")))
        for name in metrics[4:]:
            values = [r[name] for r in rows]
            print(f"  {name}: mean {mean(values):+.9f}; worst {max(values):+.9f}; "
                  f"above +0.005: {sum(v > 0.005 for v in values)}/24")
        matches = sum(r["same_winner"] for r in units if r["budget"] == budget)
        print(f"  Same validation winner: {matches}/48")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "audit"))
    parser.add_argument("--archive", type=Path, default=base.ARCHIVE)
    args = parser.parse_args()
    base.ARCHIVE = args.archive
    checksum = RUN / "freeze.sha256"
    expected = {Path(__file__).resolve(), RUN / "Protocol.md"}
    for line in checksum.read_text().splitlines():
        value, name = line.split()
        path = (base.ROOT / name).resolve()
        assert path in expected and digest(path) == value
        expected.remove(path)
    assert not expected
    earlier = multi.verify_freeze()
    archive = {(r["seed"], r["encoder"], r["domain"], r["method"]): r
               for r in multi.load_traces(earlier) if r["method"] in ("pool_only", "second_moment_mmd")}
    metadata = dict(protocol=digest(checksum), sequential=earlier,
                    sequential_decisions=digest(multi.RUN / multi.TRACES))
    print(f"NumPy {np.__version__}. Verified frozen input traces and protocol hashes.")
    (select if args.phase == "select" else audit)(metadata, archive)
