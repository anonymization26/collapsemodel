"""Audit locked 25-query pool fallback selections against archived test scores."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean

POLICIES = ("hybrid", "continue25", "switch25", "sequential25", "mmd25", "mmd50", "full455")
QUERY_LIMITS = dict.fromkeys(POLICIES, 25) | {"mmd50": 50, "full455": 455}
TOLERANCE = 0.005
METRICS = ("brier", "accuracy", "nll")
METRIC_TOLERANCES = (2e-9, 2e-9, 2e-8)
LABELS = {
    "hybrid": "检测后切换",
    "continue25": "相同探针后始终继续",
    "switch25": "相同探针后始终切换",
    "sequential25": "原逐次 25",
    "mmd25": "固定 MMD 25",
    "mmd50": "固定 MMD 50",
    "full455": "完整验证 455",
}
UNIT_COLUMNS = (
    "unit", "dataset", "partition", "domain", "seed", "triggered", "diagnostic_score",
    "probe_residuals", "method", "query_count", "winner", "validation_brier",
    "validation_regret", "brier", "accuracy", "nll", "gap_to_full", "gap_to_sequential25",
    "same_full_winner", "test_source",
)
GROUP_COLUMNS = (
    "dataset", "partition", "domain", "method", "n", "test_n", "triggered",
    "mean_validation_brier", "mean_validation_regret", "mean_test_brier",
    "mean_gap_to_full", "mean_gap_to_sequential25", "gt_005_to_full",
    "gt_005_to_sequential25", "same_full_winner",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sha(path, expected):
    if sha(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def check_selection(run):
    """Verify the seal and structural invariants before any test file is opened."""
    lock = load_json(run / "selection.lock.json")
    verify_sha(run / "selection.json", lock["selection_sha256"])
    verify_sha(run / "freeze.json", lock["freeze_sha256"])
    selections = load_json(run / "selection.json")
    units = set()
    for item in selections:
        unit = item["unit"]
        if unit in units or not unit or Path(unit).name != unit:
            raise ValueError(f"Invalid or duplicate configuration {unit!r}")
        units.add(unit)
        policies = item["policies"]
        if set(policies) != set(POLICIES):
            raise ValueError(f"Incorrect policy set in {unit}")
        validated = {}
        for policy, decision in policies.items():
            queries = decision["queries"]
            observed = decision["observed"]
            if not (len(queries) == len(set(queries)) == len(observed) == QUERY_LIMITS[policy]):
                raise ValueError(f"Invalid number of queries in {unit}/{policy}")
            for query, value in zip(queries, observed):
                if (type(query) is not int or not 0 <= query < 455 or
                        not math.isfinite(value) or not 0 <= value <= 2):
                    raise ValueError(f"Invalid validation observation in {unit}/{policy}")
                prior = validated.setdefault(query, value)
                if not math.isclose(value, prior, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(f"Inconsistent validation observation in {unit}/{query}")
            if decision["winner"] != min(zip(observed, queries))[1]:
                raise ValueError(f"Invalid validation winner in {unit}/{policy}")
        choice = "switch25" if item["triggered"] else "continue25"
        if policies["hybrid"] != policies[choice]:
            raise ValueError(f"Hybrid does not match its locked decision in {unit}")
    if not selections:
        raise ValueError("No locked configurations")
    return selections


def verified_outcomes(path, checksum_field):
    """Verify the source files bound by an archived selection and test audit."""
    audit = load_json(path / "audit.json")
    if audit["status"] != "complete":
        raise ValueError(f"Archive audit incomplete: {path}")
    lock_path = path / "selection_lock.json"
    verify_sha(lock_path, audit["selection_lock"])
    lock = load_json(lock_path)
    verify_sha(path / "freeze.json", lock["freeze"])
    verify_sha(path / "decisions.json", lock["decisions"])
    verify_sha(path / "outcomes.csv", audit[checksum_field])
    return audit, csv_rows(path / "outcomes.csv")


def collect_records(archive, selections, supplemental):
    selected = {(item["unit"], v["winner"]) for item in selections for v in item["policies"].values()}
    linear_aggregate = defaultdict(list)
    for row in csv_rows(archive / "linear-bai-20261117" / "all_outcomes.csv"):
        linear_aggregate[row["unit"]].append(row)

    tests = {}

    def insert(row, source):
        key = (row["unit"], int(row["winner"]))
        metric_values = tuple(float(row[field]) for field in METRICS)
        if (not all(math.isfinite(value) for value in metric_values) or
                not 0 <= metric_values[0] <= 2 or
                not 0 <= metric_values[1] <= 1 or metric_values[2] < 0):
            raise ValueError(f"Invalid archived test metrics for {key}")
        if key in tests:
            prior, prior_source = tests[key]
            if not all(math.isclose(x, y, rel_tol=0, abs_tol=tol) for x, y, tol in
                       zip(prior, metric_values, METRIC_TOLERANCES)):
                raise ValueError(f"Mismatched archived test scores for {key}: {prior_source} vs {source}")
        else:
            tests[key] = metric_values, source

    for item in selections:
        unit = item["unit"]
        fresh, fresh_rows = verified_outcomes(archive / "fresh-20261117" / unit, "outcomes_sha256")
        linear, linear_rows = verified_outcomes(archive / "linear-bai-20261117" / unit, "outcomes")
        if linear["old_lock"] != fresh["selection_lock"]:
            raise ValueError(f"Linear archive is bound to a different source design: {unit}")
        if linear_rows != linear_aggregate[unit]:
            raise ValueError(f"Aggregate linear outcomes differ from sealed source: {unit}")
        for source, rows in (("fresh_archive", fresh_rows), ("linear_archive", linear_rows)):
            for row in rows:
                if row["unit"] != unit:
                    raise ValueError(f"Wrong configuration in {source}: {unit}")
                insert(row, source)
    if supplemental is not None:
        for row in csv_rows(supplemental):
            key = row["unit"], int(row["winner"])
            if key not in selected:
                raise ValueError(f"Supplemental test score is outside the selection lock: {key}")
            insert(row, "supplemental_locked_models")
    return tests


def build_rows(selections, tests):
    rows = []
    missing = defaultdict(list)
    for item in selections:
        base = {field: item[field] for field in UNIT_COLUMNS[:7]}
        base["probe_residuals"] = json.dumps(item["probe_residuals"], ensure_ascii=False)
        unit = item["unit"]
        policies = item["policies"]
        full = policies["full455"]
        full_val = min(full["observed"])
        full_id = full["winner"]
        full_brier = tests[(unit, full_id)][0][0]
        sequential_id = policies["sequential25"]["winner"]
        seq_brier = tests[(unit, sequential_id)][0][0]
        for name in POLICIES:
            policy = policies[name]
            winner = policy["winner"]
            value = min(policy["observed"])
            row = base | dict(method=name, winner=winner, query_count=len(policy["queries"]),
                              validation_brier=value, validation_regret=value-full_val,
                              same_full_winner=int(winner == full_id))
            metrics, source = tests.get((unit, winner), (None, "missing"))
            row["test_source"] = source
            if metrics is None:
                missing[(unit, winner)].append(name)
            else:
                row.update(zip(METRICS, metrics))
                row.update(gap_to_full=metrics[0]-full_brier,
                           gap_to_sequential25=metrics[0]-seq_brier)
            rows.append(row)
    return rows, missing


def results_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        d, p, domain, method = (row[k] for k in ("dataset", "partition", "domain", "method"))
        group_keys = {(d, "all", "all", method), (d, p, "all", method)}
        if d == "domainnet":
            group_keys.update(((d, "all", domain, method), (d, p, domain, method)))
        for k in group_keys:
            grouped[k].append(row)

    def avg(available, field):
        return fmean(row[field] for row in available) if available else None

    results = []
    for key, items in sorted(grouped.items(), key=lambda kv: (*kv[0][:3], POLICIES.index(kv[0][3]))):
        available = [row for row in items if "brier" in row]
        result = dict(zip(GROUP_COLUMNS[:4], key))
        result.update(
            n=len(items), test_n=len(available), triggered=sum(row["triggered"] for row in items),
            mean_validation_brier=avg(items, "validation_brier"),
            mean_validation_regret=avg(items, "validation_regret"),
            mean_test_brier=avg(available, "brier"),
            mean_gap_to_full=avg(available, "gap_to_full"),
            mean_gap_to_sequential25=avg(available, "gap_to_sequential25"),
            gt_005_to_full=sum(row["gap_to_full"] > TOLERANCE for row in available) if available else None,
            gt_005_to_sequential25=(
                sum(row["gap_to_sequential25"] > TOLERANCE for row in available) if available else None
            ),
            same_full_winner=sum(row["same_full_winner"] for row in items),
        )
        results.append(result)
    return results


def markdown_summary(groups, missing, count):
    def value(x, signed=False):
        if x is None:
            return "—"
        return f"{x:+.6f}" if signed else f"{x:.6f}"

    def exceed(x, n):
        return f"{x}/{n}" if n else "—"

    doc = [
        "# 25 次试跑内的加性诊断与 MMD 切换",
        "",
        f"回放覆盖 {count} 个配置。完整 455 的参考模型由验证集选出。验证后悔值是所选模型与这个参照的验证 Brier 之差。",
        "规则在本轮读取逐候选损失前冻结。此前已知原实验的汇总结果，因此这是对既有数据的回顾性评估。",
        "测试差值严格超过 +0.005 计一次。阈值为描述性标准，各配置共享数据。",
        "",
    ]
    if missing:
        doc.append(
            f"归档只包含此前审计过的候选赢家，{len(missing)} 个锁定模型没有测试记录。"
            "测试均值和差值只使用每行已覆盖的配置，行间覆盖可能不同。所有配置均纳入验证统计。"
        )
    else:
        doc.append("所有锁定赢家都找到测试分数；选型轨迹先锁定，测试阶段只审计锁定赢家。")
    for dataset, partition in (("domainnet", "all"), ("domainnet", "hash"), ("domainnet", "pc1"),
                               ("optdigits", "hash"), ("optdigits", "pc1")):
        items = [row for row in groups if (row["dataset"], row["partition"], row["domain"]) ==
                 (dataset, partition, "all")]
        if not items:
            continue
        doc += [
            "", f"## {dataset} / {partition}", "",
            f"检出并切换 {items[0]['triggered']}/{items[0]['n']} 个配置。",
            "",
            "| 策略 | 测试覆盖 | 平均测试 Brier | Δ 对完整验证 | Δ 对原逐次 25 | 对完整 >.005 | 对原逐次 >.005 | 平均验证后悔值 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in items:
            n = row["test_n"]
            doc.append(
                f"| {LABELS[row['method']]} | {n}/{row['n']} | {value(row['mean_test_brier'])} | "
                f"{value(row['mean_gap_to_full'], True)} | {value(row['mean_gap_to_sequential25'], True)} | "
                f"{exceed(row['gt_005_to_full'], n)} | {exceed(row['gt_005_to_sequential25'], n)} | "
                f"{value(row['mean_validation_regret'])} |"
            )
    domains = [
        row for row in groups if row["dataset"] == "domainnet" and row["domain"] != "all"
        and row["partition"] == "all" and row["method"] == "hybrid"
    ]
    if domains:
        doc += [
            "", "## DomainNet 各目标域（切换策略）", "",
            "| 目标域 | 切换 | 测试覆盖 | 平均测试 Brier | Δ 对完整验证 | Δ 对原逐次 25 | 对完整 >.005 | 对原逐次 >.005 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in domains:
            n = row["test_n"]
            doc.append(
                f"| {row['domain']} | {row['triggered']}/{row['n']} | {n}/{row['n']} | "
                f"{value(row['mean_test_brier'])} | {value(row['mean_gap_to_full'], True)} | "
                f"{value(row['mean_gap_to_sequential25'], True)} | {exceed(row['gt_005_to_full'], n)} | "
                f"{exceed(row['gt_005_to_sequential25'], n)} |"
            )
    doc += [
        "", "逐配置见 [per_unit.csv](per_unit.csv)，分组汇总见 [group_summary.csv](group_summary.csv)。",
    ]
    if missing:
        doc.append("锁定后仍缺测试记录的候选见 [missing_test_winners.csv](missing_test_winners.csv)。")
    return "\n".join(doc) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--supplemental-test-csv", type=Path)
    args = parser.parse_args()
    run = args.run_dir
    selections = check_selection(run)
    tests = collect_records(args.archive, selections, args.supplemental_test_csv)
    rows, missing = build_rows(selections, tests)
    groups = results_groups(rows)

    write_csv(run / "per_unit.csv", UNIT_COLUMNS, rows)
    write_csv(run / "group_summary.csv", GROUP_COLUMNS, groups)
    write_csv(run / "missing_test_winners.csv", ("unit", "winner", "methods"), [
        dict(unit=unit, winner=winner, methods=" ".join(names))
        for (unit, winner), names in sorted(missing.items())
    ])
    (run / "Results.md").write_text(markdown_summary(groups, missing, len(selections)), encoding="utf-8")
    summary = dict(
        units=len(selections),
        policy_outcomes=len(rows),
        missing_unique_test_scores=len(missing),
        missing_policy_test_scores=sum(len(names) for names in missing.values()),
        selection_lock_sha256=sha(run / "selection.lock.json"),
        supplemental_test_sha256=sha(args.supplemental_test_csv) if args.supplemental_test_csv else None,
        outputs={name: sha(run / name) for name in
                 ("per_unit.csv", "group_summary.csv", "Results.md", "missing_test_winners.csv")},
    )
    (run / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {len(selections)} locked configurations; {len(missing)} missing unique test scores.")


if __name__ == "__main__":
    main()
