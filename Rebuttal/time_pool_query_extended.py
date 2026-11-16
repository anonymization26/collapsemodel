#!/usr/bin/env python3
"""Time four logistic policies on resident seed-20260911 DomainNet features."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import partial
import json
import platform
from pathlib import Path
from statistics import median
from threading import Barrier
from time import perf_counter

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_info, threadpool_limits

from run_pool_query import DOMAINS, ENCODERS, pool_design
from time_pool_query import (
    BUDGETS, LOGISTIC, METHODS, SCORE_ABSOLUTE_TOLERANCE, aligned_indices,
    brier_score, checked_json, digest, expected_trace, fit_readout, mmd_order,
    rank_predictions, read_json, write_json,
)


SEED = 20260911


def manifest(archive, stage, protocol_sha, encoder):
    meta = read_json(archive / stage / "manifest.json")
    assert meta["stage"] == stage and meta["status"] == "complete"
    assert meta["context"]["encoder"] == encoder
    assert meta["context"]["projection_seed"] == SEED
    assert meta["context"]["protocol_sha256"] == protocol_sha
    return meta


def load_inputs(ridge_dir, transport_dir, domain, encoder):
    ridge_manifest = read_json(ridge_dir / "temperature_inputs.json")
    screen_manifest = read_json(transport_dir / "pilot.json")
    assert ridge_manifest["encoder"] == encoder == screen_manifest["encoder"]["name"]
    samples = ridge_manifest["samples"]
    screen_samples = screen_manifest["samples"]
    with np.load(ridge_dir / "verified_features.npz", allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float64)
    screen_features = np.asarray(
        np.load(transport_dir / "transport_features.npy", allow_pickle=False), dtype=np.float64
    )
    assert features.shape == (len(samples), 128)
    assert screen_features.shape == (len(screen_samples), 128)
    rows = {sample["sample_id"]: i for i, sample in enumerate(samples)}
    screen_rows = {sample["sample_id"]: i for i, sample in enumerate(screen_samples)}
    assert len(rows) == len(samples) and len(screen_rows) == len(screen_samples)
    blocks = {block["candidate_id"]: block["sample_ids"] for block in screen_manifest["blocks"]}
    ridge_blocks = {block["candidate_id"]: block["sample_ids"]
                    for block in read_json(ridge_dir / "pilot.json")["blocks"]}
    assert blocks == ridge_blocks and len(blocks) == 15
    assert all(len(members) == len(set(members)) == 512 for members in blocks.values())
    source_ids = sorted({key for members in blocks.values() for key in members})
    assert len(source_ids) == 7680
    source = [rows[key] for key in source_ids]
    screen_source = [screen_rows[key] for key in source_ids]
    for i, j in zip(source, screen_source):
        left, right = samples[i], screen_samples[j]
        assert left["domain"] == right["domain"] != domain
        assert left["role"] == right["role"] == "anchor_pool"
    source_error = float(np.max(np.abs(features[source] - screen_features[screen_source])))
    assert source_error <= SCORE_ABSOLUTE_TOLERANCE, source_error
    labels = np.asarray([int(sample["class_id"]) for sample in samples], dtype=np.int64)
    assert labels.min() >= 0 and labels.max() < 100
    return (features, screen_features, labels, rows, screen_rows, blocks,
            aligned_indices(samples, domain, "target_validation", 1024),
            aligned_indices(samples, domain, "target_test", 2048),
            aligned_indices(screen_samples, domain, "target_selection", 1536), source_error)


def evaluate(candidate, features, labels, indices, validation_features, validation_labels):
    tick = perf_counter()
    source = np.concatenate([indices[name] for name in candidate.split("|")])
    parameters = fit_readout(features[source], labels[source])
    fit_seconds = perf_counter() - tick
    tick = perf_counter()
    loss = brier_score(validation_features, validation_labels, *parameters)
    return loss, parameters, fit_seconds, perf_counter() - tick


def run_method(method, ids, inputs, archived_mmd, executor):
    (features, screen_features, labels, rows, screen_rows, blocks,
     validation, _, target, _) = inputs
    start = perf_counter()
    indices = {name: np.array([rows[key] for key in members]) for name, members in blocks.items()}
    fit_candidate = partial(
        evaluate, features=features, labels=labels, indices=indices,
        validation_features=features[validation], validation_labels=labels[validation],
    )
    times = dict(preparation=perf_counter() - start, design=0.0, screening=0.0,
                 evaluation_wall=0.0, fit_worker_sum=0.0, validation_worker_sum=0.0,
                 acquisition=0.0)
    queries, observed, weights = [], [], []
    mmd_gate = None

    def evaluate_batch(candidates):
        tick = perf_counter()
        if executor is None or len(candidates) == 1:
            results = list(map(fit_candidate, candidates))
        else:
            results = list(executor.map(fit_candidate, candidates))
        times["evaluation_wall"] += perf_counter() - tick
        for candidate, (loss, parameters, fit_time, val_time) in zip(candidates, results):
            times["fit_worker_sum"] += fit_time
            times["validation_worker_sum"] += val_time
            queries.append(candidate)
            observed.append(loss)
            weights.append(parameters)

    if method == "mmd50":
        tick = perf_counter()
        screening_blocks = {name: screen_features[[screen_rows[key] for key in members]]
                            for name, members in blocks.items()}
        ranking = mmd_order(ids, screening_blocks, screen_features[target])
        times["screening"] = perf_counter() - tick
        mmd_gate = {"exact_ranking": ranking == archived_mmd,
                    "top50_overlap": len(set(ranking[:50]) & set(archived_mmd[:50]))}
        assert mmd_gate["top50_overlap"] == 50, mmd_gate
        evaluate_batch(ranking[:50])
    elif method == "full455":
        evaluate_batch(ids)
    else:
        tick = perf_counter()
        design, queried = pool_design(ids)
        times["design"] = perf_counter() - tick
        evaluate_batch([ids[index] for index in queried])
        if method == "batch25":
            tick = perf_counter()
            additions = rank_predictions(design, queried, observed)[:10]
            times["acquisition"] = perf_counter() - tick
            evaluate_batch([ids[index] for index in additions])
        else:
            while len(queries) < 25:
                tick = perf_counter()
                index = rank_predictions(design, queried, observed)[0]
                times["acquisition"] += perf_counter() - tick
                queried.append(index)
                evaluate_batch([ids[index]])
    best = min(range(len(queries)), key=lambda i: (observed[i], queries[i]))
    times["total"] = perf_counter() - start
    assert len(queries) == len(set(queries)) == BUDGETS[method]
    return (dict(method=method, fit_count=len(queries), seconds=times, queries=queries,
                 validation_brier=observed, winner=queries[best], mmd_alignment=mmd_gate),
            weights[best])


def time_runs(ids, inputs, archived_mmd, workers, repeats):
    runs, models = [], []
    context = ThreadPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()
    with threadpool_limits(limits=1), context as executor:
        # Starting the executor is outside the resident-feature decision interval.
        libraries = threadpool_info()
        if executor is not None:
            barrier = Barrier(workers)
            list(executor.map(lambda _: barrier.wait(), range(workers)))
        for repeat in range(repeats):
            pivot = repeat % len(METHODS)
            for method in METHODS[pivot:] + METHODS[:pivot]:
                record, model = run_method(method, ids, inputs, archived_mmd, executor)
                record["repeat"] = repeat + 1
                models.append(model)
                runs.append(record)
                print(f"{repeat + 1} {method}: {record['seconds']['total']:.3f}s, "
                      f"{record['fit_count']} fits", flush=True)
    return runs, models, libraries


def audit_validation(runs, archive, meta, name, ids, ranking, output):
    validation = checked_json(archive / "validate", meta, name)
    assert sorted(validation) == ids
    with threadpool_limits(limits=1):
        expected = {method: expected_trace(method, ids, validation, ranking) for method in METHODS}
    for record in runs:
        comparison = expected[record["method"]]
        archive_winner = min(comparison, key=lambda key: (validation[key]["brier_score"], key))
        errors = [abs(value - validation[key]["brier_score"])
                  for key, value in zip(record["queries"], record["validation_brier"])]
        record["validation_alignment"] = {
            "max_abs_brier_error": max(errors),
            "matches_reference_query_order": record["queries"] == comparison,
            "matches_reference_query_set": set(record["queries"]) == set(comparison),
            "archive_winner": archive_winner, "winner_matches": record["winner"] == archive_winner,
        }
    write_json(output / "validation_alignment.json", [r["validation_alignment"] for r in runs])
    maximum_error = max(r["validation_alignment"]["max_abs_brier_error"] for r in runs)
    assert maximum_error <= SCORE_ABSOLUTE_TOLERANCE, maximum_error


def audit_test(runs, models, inputs, archive, screen_meta, val_meta, protocol_sha, encoder, name, ids):
    meta = manifest(archive, "test-audit", protocol_sha, encoder)
    assert meta["lineage"] == val_meta["lineage"]
    assert meta["upstream"] == {"screen": screen_meta["artifact_id"], "validate": val_meta["artifact_id"]}
    outcomes = checked_json(archive / "test-audit", meta, name)
    assert sorted(outcomes) == ids
    features, _, labels, *_, test_rows, _, _ = inputs
    with threadpool_limits(limits=1):
        for record, model in zip(runs, models):
            actual = brier_score(features[test_rows], labels[test_rows], *model)
            archived = outcomes[record["winner"]]["brier_score"]
            record.update(test_brier=actual, archived_winner_test_brier=archived,
                          test_brier_abs_error=abs(actual - archived))
            assert record["test_brier_abs_error"] <= SCORE_ABSOLUTE_TOLERANCE


def summarize(runs):
    summary = {}
    for method in METHODS:
        records = [r for r in runs if r["method"] == method]
        summary[method] = dict(
            fit_count=BUDGETS[method],
            median_seconds={key: median(r["seconds"][key] for r in records)
                            for key in records[0]["seconds"]},
            winner_ids=sorted({r["winner"] for r in records}),
            test_brier=[r["test_brier"] for r in records],
            archived_winner_agreement_count=sum(r["validation_alignment"]["winner_matches"] for r in records),
            archived_query_order_agreement_count=sum(r["validation_alignment"]["matches_reference_query_order"]
                                                      for r in records),
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ridge-dir", required=True, type=Path)
    parser.add_argument("--transport-dir", required=True, type=Path)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--encoder", choices=ENCODERS, default="resnet50")
    parser.add_argument("--domain", choices=DOMAINS, default="clipart")
    parser.add_argument("--workers", type=int, choices=(1, 4), default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    protocol = args.protocol.read_bytes()
    protocol_sha = digest(protocol)
    assert json.loads(protocol)["logistic"] == LOGISTIC
    screen_meta = manifest(args.archive_dir, "screen", protocol_sha, args.encoder)
    val_meta = manifest(args.archive_dir, "validate", protocol_sha, args.encoder)
    assert val_meta["lineage"] == screen_meta["lineage"]
    assert val_meta["upstream"] == {"screen": screen_meta["artifact_id"]}
    ranks = checked_json(args.archive_dir / "screen", screen_meta, "rankings.json")
    ranking = ranks[args.domain]["second_moment_mmd"]
    ids = sorted(ranking)
    assert len(ids) == len(set(ids)) == 455
    inputs = load_inputs(args.ridge_dir, args.transport_dir, args.domain, args.encoder)
    args.output.mkdir(parents=True, exist_ok=False)
    print(f"{args.encoder}/{args.domain}/seed {SEED}: {args.workers} fit workers.", flush=True)
    runs, models, libraries = time_runs(ids, inputs, ranking, args.workers, args.repeats)
    metadata = dict(encoder=args.encoder, domain=args.domain, seed=SEED, workers=args.workers,
                    repeats=args.repeats, numpy=np.__version__, scipy=scipy.__version__,
                    sklearn=sklearn.__version__, python=platform.python_version(),
                    platform=platform.platform(), threadpools=libraries,
                    timing_scope="resident_projected_features_to_validation_winner",
                    summed_worker_times_are_nonadditive_with_wall_times=True,
                    logistic=LOGISTIC, protocol_sha256=protocol_sha, source_features_max_difference=inputs[-1])
    output = args.output
    decisions = output / "decisions.json"
    write_json(decisions, {"metadata": metadata, "runs": runs})
    decisions_sha = digest(decisions.read_bytes())
    (output / "decisions.sha256").write_text(f"{decisions_sha}  decisions.json\n")

    filename = f"{args.domain}__logistic_losses.json"
    audit_validation(runs, args.archive_dir, val_meta, filename, ids, ranking, output)
    assert digest(decisions.read_bytes()) == decisions_sha
    audit_test(runs, models, inputs, args.archive_dir, screen_meta, val_meta,
               protocol_sha, args.encoder, filename, ids)
    summary = summarize(runs)
    write_json(output / "timing.json", {"metadata": metadata, "decisions_sha256": decisions_sha,
                                        "summary": summary, "runs": runs})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
