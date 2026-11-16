"""Replay pool-contribution search, then separately audit frozen choices."""

import argparse
import csv
import hashlib
import json
import math
import platform
from itertools import combinations
from pathlib import Path
from statistics import mean

import numpy as np


REBUTTAL = Path(__file__).resolve().parent
ROOT = REBUTTAL.parent
RUN = REBUTTAL / "runs/domainnet-pool-query"
ARCHIVE = REBUTTAL / (
    "runs/replay-001/package/results/target_conditioned/"
    "revision_20260917/revision_results/e2b_readout_projection_v1"
)
LEGACY = REBUTTAL / "runs/domainnet-query-efficiency"
DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
ENCODERS = ("resnet50", "dinov2_b14")
BUDGETS = (15, 25, 50)
METHODS = ("geometry_pool", "pool_only", "second_moment_mmd", "target_a", "random")
LABELS = dict(zip(METHODS, ("Geometry + pool", "Pool only", "Fixed MMD", "Fixed A-opt", "Random")))
TOLERANCE = 0.005
EXPLORATION = 0.001


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify_freeze():
    freeze = RUN / "freeze.sha256"
    expected = {RUN / "Protocol.md", Path(__file__).resolve()}
    checked = set()
    for line in freeze.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = (ROOT / name).resolve()
        assert path in expected and sha256(path.read_bytes()) == digest, name
        checked.add(path)
    assert checked == expected
    return sha256(freeze.read_bytes())


def manifest(folder, stage):
    return json.loads((folder / stage / "manifest.json").read_text())


def checked_json(folder, meta, name):
    data = (folder / name).read_bytes()
    assert sha256(data) == meta["files"][name], folder / name
    return json.loads(data)


def loss_file(domain):
    return f"{domain}__logistic_losses.json"


class ValidationOracle:
    def __init__(self, records, ids):
        assert set(records) == set(ids)
        self._records = records

    def query(self, candidate):
        value = self._records[candidate]["brier_score"]
        assert math.isfinite(value) and 0 <= value <= 2
        return value


def choose_winners(queries, observed):
    return {str(k): min(zip(observed[:k], queries[:k]))[1] for k in BUDGETS}


def pool_design(ids):
    members = [tuple(candidate.split("|")) for candidate in ids]
    pools = sorted({pool for triple in members for pool in triple})
    assert len(pools) == 15 and set(members) == set(combinations(pools, 3))
    design = np.array([[pool in triple for pool in pools] for triple in members], dtype=float)
    residual = design.copy()
    basis = []
    for _ in pools:
        norms = np.einsum("ij,ij->i", residual, residual)
        norms[basis] = -1
        pivot = int(np.argmax(np.round(norms, 12)))
        assert norms[pivot] > 0
        basis.append(pivot)
        direction = residual[pivot] / math.sqrt(norms[pivot])
        residual -= np.outer(residual @ direction, direction)
    assert np.linalg.matrix_rank(design[basis]) == len(pools)
    return design, basis


def adaptive_trace(ids, design, basis, offset, reveal):
    rows = list(basis)
    observed = [reveal(ids[i]) for i in rows]
    while len(rows) < BUDGETS[-1]:
        x = design[rows]
        beta = np.linalg.lstsq(x, np.array(observed) - offset[rows], rcond=None)[0]
        lower = np.linalg.cholesky(x.T @ x)
        whitened = np.linalg.solve(lower, design.T)
        leverage = np.linalg.norm(whitened, axis=0)
        score = np.round(offset + design @ beta - EXPLORATION * leverage, 12)
        score[rows] = np.inf
        pick = int(np.argmin(score))
        rows.append(pick)
        observed.append(reveal(ids[pick]))
    return [ids[i] for i in rows], observed


def fixed_trace(order, reveal):
    queries = order[:BUDGETS[-1]]
    return queries, [reveal(candidate) for candidate in queries]


def trace(domain, encoder, method, repeat, queries, observed):
    assert len(set(queries)) == len(queries) == len(observed) == BUDGETS[-1]
    return dict(domain=domain, encoder=encoder, method=method, repeat=repeat,
                queries=queries, validation_brier=observed, winners=choose_winners(queries, observed))


def legacy_winners():
    decisions = json.loads((LEGACY / "frozen_decisions.json").read_text())["decisions"]
    return {(row["domain"], row["encoder"], row["method"], row["repeat"]): row
            for row in decisions if row["policy"] == "fixed50"}


def select(freeze_digest):
    decision_path = RUN / "frozen_decisions.jsonl"
    decision_hash = RUN / "selection.sha256"
    if decision_path.exists() or decision_hash.exists():
        raise SystemExit("Frozen decisions already exist. Keep the original run intact.")
    legacy = legacy_winners()
    checked = 0
    traces = []
    for encoder in ENCODERS:
        folder = ARCHIVE / f"{encoder}__20260911"
        screen = manifest(folder, "screen")
        val = manifest(folder, "validate")
        assert screen["lineage"] == val["lineage"]
        assert val["upstream"]["screen"] == screen["artifact_id"]
        orders = checked_json(folder / "screen", screen, "rankings.json")
        for domain in DOMAINS:
            ranked = orders[domain]
            ids = sorted(ranked["target_a"])
            assert len(ids) == len(set(ids)) == 455
            families = {method: [ranked[method]] for method in METHODS[2:4]}
            families["random"] = [ranked[f"random:{i}"] for i in range(20)]
            assert all(sorted(order) == ids for orders_ in families.values() for order in orders_)
            oracle = ValidationOracle(checked_json(folder / "validate", val, loss_file(domain)), ids)
            design, basis = pool_design(ids)
            positions = [{candidate: i for i, candidate in enumerate(ranked[method])}
                         for method in METHODS[2:4]]
            geometry = TOLERANCE * np.array([sum(r[c] for r in positions) for c in ids]) / (2 * (len(ids) - 1))
            adaptive_runs = [
                trace(domain, encoder, method, 0,
                      *adaptive_trace(ids, design, basis, offset, oracle.query))
                for method, offset in (("geometry_pool", geometry), ("pool_only", np.zeros(len(ids))))
            ]
            assert adaptive_runs[0]["queries"][:15] == adaptive_runs[1]["queries"][:15]
            traces.extend(adaptive_runs)
            for method, orders_ in families.items():
                for repeat, order in enumerate(orders_):
                    row = trace(domain, encoder, method, repeat, *fixed_trace(order, oracle.query))
                    old = legacy[domain, encoder, method, repeat]
                    assert old["queries"] == 50 and row["winners"]["50"] == old["winner"]
                    checked += 1
                    traces.append(row)
            print(f"PASS {domain}/{encoder}: 455 candidates, 15-pool design condition "
                  f"{np.linalg.cond(design[basis]):.4f}, 24 traces.")
    assert checked == len(legacy) == 264 and len(traces) == 288
    with decision_path.open("x") as stream:
        stream.write(json.dumps({"protocol_freeze_sha256": freeze_digest}) + "\n")
        for row in traces:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    with decision_hash.open("x") as stream:
        stream.write(f"{sha256(decision_path.read_bytes())}  {decision_path.name}\n")
    print(f"PASS: {checked} fixed-50 baseline winners match the prior frozen replay.")
    print(f"PASS: {len(traces)} traces, {len(traces) * 50} query records and "
          f"{len(traces) * len(BUDGETS)} winners locked. Test audit files were not opened.")


def frozen_traces(freeze_digest):
    path = RUN / "frozen_decisions.jsonl"
    digest, name = (RUN / "selection.sha256").read_text().split()
    data = path.read_bytes()
    assert name == path.name and sha256(data) == digest
    header, *rows = [json.loads(line) for line in data.splitlines()]
    assert header["protocol_freeze_sha256"] == freeze_digest and len(rows) == 288
    return rows


def write_csv(name, rows):
    with (RUN / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(trials):
    domain_rows, overall = [], []
    metrics = ("test_brier", "full_test_brier", "delta")
    for budget in BUDGETS:
        for method in METHODS:
            rows = []
            for domain in DOMAINS:
                unit = {encoder: [r for r in trials if (r["budget"], r["method"], r["domain"], r["encoder"])
                                  == (budget, method, domain, encoder)] for encoder in ENCODERS}
                assert all(len(group) == (20 if method == "random" else 1) for group in unit.values())
                stats = {k: mean(mean(r[k] for r in group) for group in unit.values()) for k in metrics}
                rows.append(dict(budget=budget, method=method, domain=domain, **stats))
            domain_rows.extend(rows)
            overall.append(dict(budget=budget, method=method,
                                **{k: mean(r[k] for r in rows) for k in metrics},
                                domains_over_005=sum(r["delta"] > TOLERANCE for r in rows)))
    return domain_rows, overall


def report(domain_rows, overall):
    by_domain = {(r["budget"], r["method"], r["domain"]): r for r in domain_rows}
    contrast = []
    for domain in DOMAINS:
        mmd = by_domain[50, "second_moment_mmd", domain]["test_brier"]
        geometry = by_domain[25, "geometry_pool", domain]["test_brier"]
        pool = by_domain[25, "pool_only", domain]["test_brier"]
        contrast.append(dict(domain=domain, geometry_25=geometry, pool_only_25=pool, mmd_50=mmd,
                             geometry_gap=geometry - mmd, pool_only_gap=pool - mmd))
    write_csv("primary_contrast.csv", contrast)
    keys = ("geometry_25", "pool_only_25", "mmd_50", "geometry_gap", "pool_only_gap")
    average = {k: mean(r[k] for r in contrast) for k in keys}
    geo_tail = sum(r["geometry_gap"] > TOLERANCE for r in contrast)
    pool_tail = sum(r["pool_only_gap"] > TOLERANCE for r in contrast)
    lines = [
        "# DomainNet sequential pool-contribution results", "",
        "Archived logistic readout at projection seed 20260911, six targets and two encoders. "
        "Each method uses only its own queried validation scores. Test scores were read after decisions were locked. "
        "Δ is the selected model's test Brier minus the test Brier of the 455-fit validation winner. "
        "Lower is better. Random averages 20 orders per unit before averaging encoders and targets.", "",
        "## Primary contrast against fixed MMD at 50", "",
        "| Target | Geometry + pool 25 | Pool only 25 | MMD 50 | Geometry gap | Pool-only gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in [*contrast, dict(domain="Mean", **average)]:
        lines.append(f"| {row['domain']} | {row['geometry_25']:.6f} | {row['pool_only_25']:.6f} | "
                     f"{row['mmd_50']:.6f} | {row['geometry_gap']:+.6f} | {row['pool_only_gap']:+.6f} |")
    lines.extend(["", f"The signed gap versus MMD 50 exceeds +0.005 in {geo_tail}/6 domains for geometry + pool "
                  f"and {pool_tail}/6 for pool only.", "", "## Overall against exhaustive validation", "",
                  f"Mean test Brier of the exhaustive-validation reference: {overall[0]['full_test_brier']:.6f}.", "",
                  "| Budget | Method | Test Brier | Δ | Domains Δ > 0.005 |", "|---:|---|---:|---:|---:|"])
    for row in overall:
        lines.append(f"| {row['budget']} | {LABELS[row['method']]} | {row['test_brier']:.6f} | "
                     f"{row['delta']:+.6f} | {row['domains_over_005']}/6 |")
    for budget in BUDGETS:
        lines.extend(["", f"## Per-domain results at {budget} queries", "",
                      "Each method cell is test Brier (Δ to exhaustive validation).", "",
                      "| Target | Full | " + " | ".join(LABELS.values()) + " |", "|---|---:|---:|---:|---:|---:|---:|"])
        for domain in DOMAINS:
            rows = [by_domain[budget, method, domain] for method in METHODS]
            cells = [f"{r['test_brier']:.6f} ({r['delta']:+.6f})" for r in rows]
            lines.append(f"| {domain} | {rows[0]['full_test_brier']:.6f} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Scope", "",
                  "The adaptive methods have identical choices at budget 15 because their exploration prefix is shared. "
                  "The full reference is a validation winner, so a negative test gap can occur. The +0.005 counts use "
                  "domain means and a signed tolerance. The targets share source pools. "
                  "The existing archive was used in earlier analyses. Query counts describe simulated fit budgets."])
    (RUN / "Results.md").write_text("\n".join(lines) + "\n")


def audit(freeze_digest):
    traces = frozen_traces(freeze_digest)
    legacy = legacy_winners()
    print("PASS: decision file matches the pre-audit SHA-256 and frozen protocol.")
    units = {}
    for encoder in ENCODERS:
        folder = ARCHIVE / f"{encoder}__20260911"
        screen, val = (manifest(folder, stage) for stage in ("screen", "validate"))
        for domain in DOMAINS:
            loss = checked_json(folder / "validate", val, loss_file(domain))
            full = min(loss, key=lambda c: (loss[c]["brier_score"], c))
            units[domain, encoder] = folder, screen, val, loss, full
    for row in traces:
        key = row["domain"], row["encoder"], row["method"], row["repeat"]
        _, _, _, loss, full = units[key[:2]]
        assert len(set(row["queries"])) == len(row["queries"]) == len(row["validation_brier"]) == 50
        assert row["validation_brier"] == [loss[c]["brier_score"] for c in row["queries"]]
        assert row["winners"] == choose_winners(row["queries"], row["validation_brier"])
        if key in legacy:
            assert full == legacy[key]["full_winner"]
    print("PASS: all query scores, prefix winners and legacy full-validation IDs verified before test audit.")
    test = {}
    for unit, (folder, screen, val, loss, _) in units.items():
        meta = manifest(folder, "test-audit")
        assert meta["lineage"] == screen["lineage"] == val["lineage"]
        assert meta["upstream"]["screen"] == screen["artifact_id"]
        assert meta["upstream"]["validate"] == val["artifact_id"]
        scores = checked_json(folder / "test-audit", meta, loss_file(unit[0]))
        assert len(scores) == 455 and set(scores) == set(loss)
        test[unit] = scores
    trials = []
    for row in traces:
        unit = row["domain"], row["encoder"]
        full = units[unit][-1]
        full_score = test[unit][full]["brier_score"]
        for budget in BUDGETS:
            winner = row["winners"][str(budget)]
            score = test[unit][winner]["brier_score"]
            trials.append(dict(domain=unit[0], encoder=unit[1], method=row["method"], repeat=row["repeat"],
                               budget=budget, winner=winner, full_winner=full,
                               test_brier=score, full_test_brier=full_score, delta=score - full_score))
    domain_rows, overall = summarize(trials)
    with (LEGACY / "per_domain.csv").open() as stream:
        old = {(r["domain"], r["method"]): r for r in csv.DictReader(stream) if r["policy"] == "fixed50"}
    matched = 0
    for row in domain_rows:
        key = row["domain"], row["method"]
        if row["budget"] == 50 and key in old:
            for field in ("test_brier", "full_test_brier", "delta"):
                assert math.isclose(row[field], float(old[key][field]), abs_tol=1e-12, rel_tol=0), (key, field)
            matched += 1
    assert matched == len(old) == 18
    print("PASS: 12 test files match their hashes and candidate IDs, and all 18 fixed-50 domain baselines reproduce.")
    write_csv("per_trial.csv", trials)
    write_csv("per_domain.csv", domain_rows)
    write_csv("overall.csv", overall)
    report(domain_rows, overall)
    print(f"PASS: wrote {len(trials)} trial rows, {len(domain_rows)} domain rows and {len(overall)} summaries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "audit"))
    args = parser.parse_args()
    print(f"Phase {args.phase}; Python {platform.python_version()}; NumPy {np.__version__}.")
    digest = verify_freeze()
    print(f"PASS: protocol and code hashes verified. Freeze SHA-256 {digest}.")
    (select if args.phase == "select" else audit)(digest)
