#!/usr/bin/env python3
"""Verify and summarize the twelve-unit resident-feature logistic timing study."""

import argparse
import csv
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean, median

ENCODERS = ("resnet50", "dinov2_b14")
DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
WORKERS = (1, 4)
METHODS = ("sequential25", "batch25", "mmd50", "full455")
FIT_COUNTS = dict(zip(METHODS, (25, 25, 50, 455)))
PHASES = ("preparation", "design", "screening", "evaluation_wall", "fit_worker_sum",
          "validation_worker_sum", "acquisition", "total")
REPEATS = 3
SCORE_TOLERANCE = 1e-5
PRACTICAL_MARGIN = 0.005
PROTOCOL_SHA = "2d6d07ba5d2071f6d5a64285881f87c823a2ea634431b8f93e3711eda7e99cb9"
LOGISTIC = {"C": 1.0, "solver": "lbfgs", "fit_intercept": False, "max_iter": 1000, "tol": 1e-6}


def load_json(path):
    return json.loads(path.read_bytes())


def checksum(raw):
    return sha256(raw).hexdigest()


def check(condition, unit, label):
    if not condition:
        raise ValueError(f"{unit}: {label}")


def runner_order():
    return [(rep + 1, method)
            for rep in range(REPEATS)
            for method in METHODS[rep:] + METHODS[:rep]]


def validate_run(run, frozen_run, alignment, unit, candidates):
    mutable = {"validation_alignment", "test_brier", "archived_winner_test_brier", "test_brier_abs_error"}
    check({key: value for key, value in run.items() if key not in mutable} == frozen_run,
          unit, "timing alters frozen decisions")
    method = run["method"]
    count = FIT_COUNTS[method]
    query = run["queries"]
    brier = run["validation_brier"]
    check(run["fit_count"] == len(query) == len(set(query)) == len(brier) == count,
          unit, "fit counts or queries are inconsistent")
    check(set(query) <= candidates and all(math.isfinite(x) and 0 <= x <= 2 for x in brier),
          unit, "invalid query or validation loss")
    check(run["winner"] == min(zip(brier, query))[1], unit, "winner is not a validation minimizer")
    if method == "mmd50":
        check(run["mmd_alignment"]["top50_overlap"] == 50, unit, "MMD top-50 mismatch")
    check(alignment == run["validation_alignment"], unit, "validation alignment artifact mismatch")
    check(alignment["archive_winner"] in candidates and
          0 <= alignment["max_abs_brier_error"] <= SCORE_TOLERANCE and
          alignment["winner_matches"] == (alignment["archive_winner"] == run["winner"]),
          unit, "invalid validation alignment")
    check(not alignment["matches_reference_query_order"] or alignment["matches_reference_query_set"],
          unit, "query order match without set match")
    test_error = abs(run["test_brier"] - run["archived_winner_test_brier"])
    check(math.isclose(test_error, run["test_brier_abs_error"], abs_tol=1e-15, rel_tol=0) and
          test_error <= SCORE_TOLERANCE and 0 <= run["test_brier"] <= 2,
          unit, "test-score audit failed")
    times = run["seconds"]
    check(set(times) == set(PHASES) and
          all(math.isfinite(value) and value >= 0 for value in times.values()) and times["total"] > 0,
          unit, "invalid phase timing")
    wall_phases = ("preparation", "design", "screening", "evaluation_wall", "acquisition")
    check(sum(times[key] for key in wall_phases) <= times["total"] + 1e-9,
          unit, "wall phases exceed total time")


def verify_unit(folder, encoder, domain, workers):
    unit = f"{encoder}/{domain}/workers-{workers}"
    timing = load_json(folder / "timing.json")
    metadata, runs = timing["metadata"], timing["runs"]
    frozen_raw = (folder / "decisions.json").read_bytes()
    decision_hash = checksum(frozen_raw)
    check((folder / "decisions.sha256").read_text().split() == [decision_hash, "decisions.json"]
          and timing["decisions_sha256"] == decision_hash, unit, "decision SHA-256 mismatch")
    frozen = json.loads(frozen_raw)
    check(frozen["metadata"] == metadata, unit, "frozen metadata mismatch")
    identity_keys = ("encoder", "domain", "seed", "workers", "repeats", "protocol_sha256")
    correct = (encoder, domain, 20260911, workers, REPEATS, PROTOCOL_SHA)
    check(tuple(metadata[key] for key in identity_keys) == correct, unit, "run identity mismatch")
    check(metadata["logistic"] == LOGISTIC and
          metadata["timing_scope"] == "resident_projected_features_to_validation_winner" and
          metadata["summed_worker_times_are_nonadditive_with_wall_times"] is True and
          0 <= metadata["source_features_max_difference"] <= SCORE_TOLERANCE,
          unit, "incompatible configuration or feature caches")
    libraries = metadata["threadpools"]
    check(libraries and all(pool["num_threads"] == 1 for pool in libraries), unit, "thread limit mismatch")
    check([(run["repeat"], run["method"]) for run in runs] == runner_order(),
          unit, "repetition rotation differs")
    alignment = load_json(folder / "validation_alignment.json")
    check(len(runs) == len(frozen["runs"]) == len(alignment) == REPEATS * len(METHODS),
          unit, "incomplete audit")
    by_repeat = {(run["repeat"], run["method"]): run for run in runs}
    ordered = by_repeat[1, "full455"]["queries"]
    candidates = set(ordered)
    check(ordered == sorted(candidates) and len(ordered) == 455, unit, "invalid full archive order")
    for run, locked, align in zip(runs, frozen["runs"], alignment):
        validate_run(run, locked, align, unit, candidates)
        seq = by_repeat[run["repeat"], "sequential25"]
        batch = by_repeat[run["repeat"], "batch25"]
        check(seq["queries"][:15] == batch["queries"][:15], unit, "pool starts differ")
        if run["method"] == "full455":
            check(run["queries"] == ordered, unit, "full evaluation candidate set changed")
    check(set(timing["summary"]) == set(METHODS), unit, "missing summary methods")

    output = []
    for method in METHODS:
        records = [run for run in runs if run["method"] == method]
        calculated = {
            "fit_count": FIT_COUNTS[method],
            "median_seconds": {key: median(run["seconds"][key] for run in records) for key in PHASES},
            "winner_ids": sorted({run["winner"] for run in records}),
            "test_brier": [run["test_brier"] for run in records],
            "archived_winner_agreement_count": sum(run["validation_alignment"]["winner_matches"]
                                                   for run in records),
            "archived_query_order_agreement_count": sum(
                run["validation_alignment"]["matches_reference_query_order"] for run in records
            ),
        }
        check(timing["summary"][method] == calculated, unit, f"{method} summary differs from raw runs")
        times = calculated["median_seconds"]
        output.append(dict(encoder=encoder, domain=domain, workers=workers, method=method,
                           fits=FIT_COUNTS[method], test_brier=mean(calculated["test_brier"]),
                           **{f"{key}_seconds": value for key, value in times.items()},
                           archived_query_agreement=calculated["archived_query_order_agreement_count"],
                           archived_winner_agreement=calculated["archived_winner_agreement_count"],
                           unique_winners=len(calculated["winner_ids"]),
                           winner_ids=";".join(calculated["winner_ids"]),
                           maximum_validation_error=max(
                               r["validation_alignment"]["max_abs_brier_error"] for r in records
                           ),
                           maximum_test_error=max(r["test_brier_abs_error"] for r in records)))
    quality_rows = {(run["repeat"], run["method"]): (run["queries"], run["winner"], run["test_brier"])
                    for run in runs}
    return output, quality_rows, metadata


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze(root):
    expected = {Path(encoder, domain, f"workers-{workers}")
                for encoder, domain, workers in product(ENCODERS, DOMAINS, WORKERS)}
    actual = {path.parent.relative_to(root) for path in root.rglob("timing.json")}
    check(expected == actual, root,
          f"timing directories differ: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    by_unit, results = {}, []
    software = set()
    for encoder, domain, workers in product(ENCODERS, DOMAINS, WORKERS):
        unit_dir = root / encoder / domain / f"workers-{workers}"
        unit_rows, quality, metadata = verify_unit(unit_dir, encoder, domain, workers)
        results.extend(unit_rows)
        by_unit[encoder, domain, workers] = quality
        software.add(tuple(metadata[key] for key in ("python", "numpy", "scipy", "sklearn", "platform")))
    check(len(software) == 1, root, "software or platform differs across runs")

    lookup = {(row["encoder"], row["domain"], row["workers"], row["method"]): row for row in results}
    parallel, aggregate, domains = [], [], []
    for row in results:
        key = (row["encoder"], row["domain"], row["workers"])
        full, mmd = lookup[key + ("full455",)], lookup[key + ("mmd50",)]
        row["gap_vs_full"] = row["test_brier"] - full["test_brier"]
        row["gap_vs_mmd50"] = row["test_brier"] - mmd["test_brier"]
        row["speedup_vs_full"] = full["total_seconds"] / row["total_seconds"]
        row["speedup_vs_mmd50"] = mmd["total_seconds"] / row["total_seconds"]
    for encoder, domain, method in product(ENCODERS, DOMAINS, METHODS):
        left, right = (lookup[encoder, domain, workers, method] for workers in WORKERS)
        serial, parallel_runs = (by_unit[encoder, domain, workers] for workers in WORKERS)
        pairs = [(serial[rep, method], parallel_runs[rep, method])
                 for rep in range(1, REPEATS + 1)]
        parallel.append(dict(encoder=encoder, domain=domain, method=method,
                             seconds_1=left["total_seconds"], seconds_4=right["total_seconds"],
                             speedup_4_vs_1=left["total_seconds"] / right["total_seconds"],
                             query_order_agreement=sum(a[0] == b[0] for a, b in pairs),
                             winner_agreement=sum(a[1] == b[1] for a, b in pairs),
                             max_absolute_brier_gap=max(abs(a[2] - b[2]) for a, b in pairs),
                             gap_brier_4_minus_1=right["test_brier"] - left["test_brier"]))
    for workers, method in product(WORKERS, METHODS):
        group = [row for row in results if (row["workers"], row["method"]) == (workers, method)]
        group_domains = []
        for domain in DOMAINS:
            pair = [row for row in group if row["domain"] == domain]
            metrics = ("test_brier", "gap_vs_full", "gap_vs_mmd50", "total_seconds")
            group_domains.append(dict(domain=domain, workers=workers, method=method,
                                      **{key: mean(row[key] for row in pair) for key in metrics}))
        domains.extend(group_domains)
        schedule = [row["total_seconds"] for row in group]
        aggregate.append(dict(
            workers=workers, method=method, fits=FIT_COUNTS[method],
            mean_seconds=mean(schedule), median_seconds=median(schedule),
            min_seconds=min(schedule), max_seconds=max(schedule),
            median_speedup_vs_full=median(row["speedup_vs_full"] for row in group),
            median_speedup_vs_mmd50=median(row["speedup_vs_mmd50"] for row in group),
            mean_test_brier=mean(row["test_brier"] for row in group_domains),
            mean_gap_vs_full=mean(row["gap_vs_full"] for row in group_domains),
            mean_gap_vs_mmd50=mean(row["gap_vs_mmd50"] for row in group_domains),
            unit_gaps_vs_full_over_005=sum(row["gap_vs_full"] > PRACTICAL_MARGIN for row in group),
            domain_gaps_vs_full_over_005=sum(row["gap_vs_full"] > PRACTICAL_MARGIN for row in group_domains),
            unit_gaps_vs_mmd50_over_005=sum(row["gap_vs_mmd50"] > PRACTICAL_MARGIN for row in group),
            domain_gaps_vs_mmd50_over_005=sum(row["gap_vs_mmd50"] > PRACTICAL_MARGIN for row in group_domains),
            archived_query_agreement=sum(row["archived_query_agreement"] for row in group),
            archived_winner_agreement=sum(row["archived_winner_agreement"] for row in group),
            maximum_validation_error=max(row["maximum_validation_error"] for row in group),
            maximum_test_error=max(row["maximum_test_error"] for row in group),
        ))
    summary_parallel = []
    for method in METHODS:
        rows = [row for row in parallel if row["method"] == method]
        speedups = [row["speedup_4_vs_1"] for row in rows]
        summary_parallel.append(dict(method=method, median_speedup_4_vs_1=median(speedups),
                                     min_speedup_4_vs_1=min(speedups), max_speedup_4_vs_1=max(speedups),
                                     query_order_agreement=sum(row["query_order_agreement"] for row in rows),
                                     winner_agreement=sum(row["winner_agreement"] for row in rows),
                                     max_absolute_brier_gap=max(row["max_absolute_brier_gap"] for row in rows)))
    comparisons = []
    for workers in WORKERS:
        pairs = [(lookup[encoder, domain, workers, "sequential25"], lookup[encoder, domain, workers, "batch25"])
                 for encoder, domain in product(ENCODERS, DOMAINS)]
        ratios = [seq["total_seconds"] / batch["total_seconds"] for seq, batch in pairs]
        gaps = [batch["test_brier"] - seq["test_brier"] for seq, batch in pairs]
        domain_gaps = [mean(gaps[i:i + len(DOMAINS):len(DOMAINS)]) for i in range(len(DOMAINS))]
        same_winners = sum(
            by_unit[encoder, domain, workers][rep, "sequential25"][1] ==
            by_unit[encoder, domain, workers][rep, "batch25"][1]
            for encoder, domain, rep in product(ENCODERS, DOMAINS, range(1, REPEATS + 1))
        )
        comparisons.append(dict(workers=workers, median_speedup_batch_vs_seq=median(ratios),
                                min_speedup_batch_vs_seq=min(ratios), max_speedup_batch_vs_seq=max(ratios),
                                mean_brier_gap_batch_minus_seq=mean(gaps),
                                units_gap_over_005=sum(gap > PRACTICAL_MARGIN for gap in gaps),
                                domains_gap_over_005=sum(gap > PRACTICAL_MARGIN for gap in domain_gaps),
                                winner_agreement=same_winners))
    return results, domains, aggregate, parallel, summary_parallel, comparisons, next(iter(software))


def render(aggregate, parallel, comparisons, software):
    python, numpy, scipy, sklearn, platform = software
    lines = [
        "# DomainNet resident-feature logistic timing", "",
        "Six target domains and two encoders at projection seed 20260911. Each unit has three independent",
        "runs of each policy in rotated order at one and four workers. Within a policy run, each candidate",
        "is fitted independently. Timings start at resident projected features and end at the validation winner.",
        "Four-worker runs dispatch fitting tasks to four threads sharing resident data, with one BLAS thread",
        "per fit. Sequential acquisition fits its",
        "first 15 candidates in parallel and then fits each dependent candidate in order. Batch acquisition",
        "fits its initial 15 and remaining 10 in two separate waves. MMD50 and Full455 each fit one wave.",
        "Worker setup, feature extraction, and test scoring are outside the timed interval.", "",
        f"Environment: Python {python}, NumPy {numpy}, SciPy {scipy}, scikit-learn {sklearn}, {platform}.", "",
        "Medians are taken across repeats before aggregating across encoder and domain units. Brier scores",
        "average repetitions and encoders within each domain, then weight the six domains equally.",
        "Ratios are medians of paired unit ratios. Counts use a strictly greater than +0.005 signed gap.", "",
        "| Workers | Policy | Fits | Median seconds [unit range] | Speedup vs full | Speedup vs MMD50 | Test Brier | Gap vs full | Gap vs MMD50 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append("| {workers} | {method} | {fits} | {median_seconds:.3f} [{min_seconds:.3f}, {max_seconds:.3f}] | "
                     "{median_speedup_vs_full:.2f}× | {median_speedup_vs_mmd50:.2f}× | "
                     "{mean_test_brier:.6f} | {mean_gap_vs_full:+.6f} | {mean_gap_vs_mmd50:+.6f} |".format(**row))
    lines.extend([
        "",
        "| Workers | Policy | Gap vs full > +0.005 (domains, encoder–domains) | Gap vs MMD50 > +0.005 (domains, encoder–domains) |",
        "|---:|---|---:|---:|",
    ])
    for row in aggregate:
        lines.append("| {workers} | {method} | {domain_gaps_vs_full_over_005}/6, {unit_gaps_vs_full_over_005}/12 | "
                     "{domain_gaps_vs_mmd50_over_005}/6, {unit_gaps_vs_mmd50_over_005}/12 |".format(**row))
    lines.extend([
        "", "| Policy | Median four-worker speedup [unit range] | Same query order | Same winner | Maximum absolute Brier change |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in parallel:
        lines.append("| {method} | {median_speedup_4_vs_1:.2f}× [{min_speedup_4_vs_1:.2f}, {max_speedup_4_vs_1:.2f}] | "
                     "{query_order_agreement}/36 | {winner_agreement}/36 | {max_absolute_brier_gap:.3g} |".format(**row))
    lines.extend([
        "", "| Workers | Median batch speedup vs sequential [unit range] | Batch minus sequential Brier | Domains > +0.005 | Same winner |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in comparisons:
        lines.append("| {workers} | {median_speedup_batch_vs_seq:.2f}× [{min_speedup_batch_vs_seq:.2f}, "
                     "{max_speedup_batch_vs_seq:.2f}] | {mean_brier_gap_batch_minus_seq:+.6f} | "
                     "{domains_gap_over_005}/6 | {winner_agreement}/36 |".format(**row))
    validation_error = max(row["maximum_validation_error"] for row in aggregate)
    test_error = max(row["maximum_test_error"] for row in aggregate)
    query_matches = sum(row["archived_query_agreement"] for row in aggregate)
    winner_matches = sum(row["archived_winner_agreement"] for row in aggregate)
    lines.extend([
        "", f"All 24 unit decision files passed identity and SHA-256 checks. All 288 timed runs passed the "
        f"archive score gates. Maximum validation and test score errors were {validation_error:.3g} and "
        f"{test_error:.3g}. Archived query orders matched {query_matches}/288 and archived winners matched "
        f"{winner_matches}/288. See the per-unit tables for any differences.", "",
        "The per-unit table reports evaluation phase wall time separately from summed worker fit and validation",
        "times. Summed worker times can overlap and do not add to the wall time.", "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path, help="Server logistic-extended directory")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    units, domains, summary, parallel, fast, comparisons, software = analyze(args.input_root)
    args.output.mkdir(parents=True, exist_ok=True)
    tables = {
        "logistic_extended_per_unit.csv": units,
        "logistic_extended_per_domain.csv": domains,
        "logistic_extended_summary.csv": summary,
        "logistic_extended_parallel.csv": parallel,
        "logistic_extended_parallel_summary.csv": fast,
        "logistic_extended_pool_comparison.csv": comparisons,
    }
    for filename, rows in tables.items():
        write_csv(args.output / filename, rows)
    (args.output / "Logistic_Extended_Results.md").write_text(render(summary, fast, comparisons, software))
    print(f"Verified all 24 unit outputs and 288 timed runs. Wrote {len(tables)} CSVs and a Markdown report.")


if __name__ == "__main__":
    main()
