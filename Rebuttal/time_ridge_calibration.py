#!/usr/bin/env python3
"""Time independent ridge decisions from original resident projected features."""

import argparse
import csv
import hashlib
import json
import math
import platform
import random
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy
from scipy.special import softmax
from threadpoolctl import threadpool_info, threadpool_limits


BUDGETS = (50, 227, 455)
ARMS = ("split_uncalibrated", "split_calibrated")
TEMPERATURES = np.geomspace(0.01, 100, 81)
CLASSES = 100
STAGES = ("block_statistics_seconds", "ridge_solve_seconds", "validation_logits_seconds",
          "temperature_search_seconds", "validation_selection_seconds", "decision_seconds")


def softmax_brier(logits, labels, temperature):
    probabilities = softmax(logits / temperature, axis=1)
    return float(np.mean(np.sum(probabilities ** 2, axis=1) + 1
                         - 2 * probabilities[np.arange(len(labels)), labels]))


def validation_split(samples):
    indices = [i for i, sample in enumerate(samples) if sample["role"] == "target_validation"]
    if len(indices) != 1024:
        raise ValueError("Expected 1024 archived validation rows")

    def key(position):
        sample_id = samples[indices[position]]["sample_id"]
        return hashlib.sha256(f"ridge-temperature-v1:{sample_id}".encode()).digest(), sample_id

    ordered = sorted(range(len(indices)), key=key)
    return np.asarray(indices), ordered[::2], ordered[1::2]


def load_unit(folder):
    started = perf_counter()
    audit = json.loads((folder / "temperature_inputs.json").read_text())
    spec = json.loads((folder / "pilot.json").read_text())
    with np.load(folder / "verified_features.npz", allow_pickle=False) as cache:
        features = np.asarray(cache["features"], dtype=np.float64)
    samples = audit["samples"]
    if len(samples) != 10752 or features.shape != (len(samples), 128) or not np.isfinite(features).all():
        raise ValueError("Feature rows do not match the original unit")
    labels = np.array([int(sample["class_id"]) for sample in samples])
    if not np.all((0 <= labels) & (labels < CLASSES)):
        raise ValueError("Unexpected class ID")
    index = {sample["sample_id"]: i for i, sample in enumerate(samples)}
    if len(index) != len(samples) or len(spec["blocks"]) != 15:
        raise ValueError("Duplicate sample IDs or wrong number of source pools")
    block_rows = {block["candidate_id"]: [index[s] for s in block["sample_ids"]]
                  for block in spec["blocks"]}
    validation_rows, calibration, selection = validation_split(samples)
    test_rows = [i for i, sample in enumerate(samples) if sample["role"] == "target_test"]
    if len(test_rows) != 2048 or any(len(rows) != 512 for rows in block_rows.values()):
        raise ValueError("Unexpected source or test split size")
    candidates = sorted(audit["validation"])
    orders = audit["rankings"]
    if len(candidates) != 455 or any(sorted(ranking) != candidates for ranking in orders.values()):
        raise ValueError("Rankings do not match the frozen candidate set")
    for candidate in candidates:
        names = candidate.split("|")
        if len(set(names)) != 3 or any(name not in block_rows for name in names):
            raise ValueError(f"Invalid candidate: {candidate}")
    gate = json.loads((folder / "t1_gate.json").read_text())
    if set(gate) != {"validation", "test"} or any(not 0 <= value <= 1e-4 for value in gate.values()):
        raise ValueError("Prior T=1 reconstruction gate was not passed")
    with (folder / "results.csv").open(newline="") as source:
        expected = {(int(row["budget"]), row["method"], row["arm"]): row
                    for row in csv.DictReader(source)}
    return (features, labels, block_rows, validation_rows, calibration, selection, test_rows,
            {**orders, "full": candidates}, expected, gate, perf_counter() - started)


def decide(features, labels, block_rows, validation_rows, calibration, selection, candidates, arm):
    total_started = perf_counter()
    times = {stage: 0.0 for stage in STAGES}
    tick = perf_counter()
    validation_x, validation_y = features[validation_rows], labels[validation_rows]
    selection_y = validation_y[selection]
    calibration_y = validation_y[calibration]
    prior = np.eye(features.shape[1])
    one_hot = np.eye(CLASSES)
    times["block_statistics_seconds"] += perf_counter() - tick
    grams, rhs = {}, {}
    winner = None
    for candidate in candidates:
        names = candidate.split("|")
        tick = perf_counter()
        for name in names:
            if name not in grams:
                rows = block_rows[name]
                x = features[rows]
                grams[name] = x.T @ x
                rhs[name] = x.T @ one_hot[labels[rows]]
        times["block_statistics_seconds"] += perf_counter() - tick
        tick = perf_counter()
        gram = prior + sum(grams[name] for name in names)
        weights = np.linalg.solve((gram + gram.T) / 2, sum(rhs[name] for name in names))
        times["ridge_solve_seconds"] += perf_counter() - tick
        tick = perf_counter()
        logits = validation_x @ weights
        times["validation_logits_seconds"] += perf_counter() - tick
        temperature = 1.0
        if arm == "split_calibrated":
            tick = perf_counter()
            z = logits[calibration]
            temperature = float(min(TEMPERATURES, key=lambda t: (
                softmax_brier(z, calibration_y, t), abs(np.log(t)))))
            times["temperature_search_seconds"] += perf_counter() - tick
        tick = perf_counter()
        value = softmax_brier(logits[selection], selection_y, temperature)
        if winner is None or (value, candidate) < winner[:2]:
            winner = (value, candidate, temperature, weights)
        times["validation_selection_seconds"] += perf_counter() - tick
    times["decision_seconds"] = perf_counter() - total_started
    return winner, times


def run_unit(folder, unit, methods, repeats):
    (features, labels, block_rows, validation_rows, calibration, selection, test_rows,
     orders, expected, gate, input_seconds) = load_unit(folder)
    jobs = [(budget, method, arm) for budget in BUDGETS
            for method in (methods if budget < 455 else ("full",)) for arm in ARMS]
    records, pending = [], []
    for repeat in range(repeats):
        order = jobs.copy()
        random.Random(45649 + repeat).shuffle(order)
        for budget, method, arm in order:
            (validation_brier, winner, temperature, weights), times = decide(
                features, labels, block_rows, validation_rows, calibration, selection,
                orders[method][:budget], arm)
            check = expected[budget, method, arm]
            if winner != check["candidate"] or not math.isclose(
                    temperature, float(check["temperature"]), rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Winner/temperature mismatch: {unit}/{budget}/{method}/{arm}")
            row = dict(unit=unit, budget=budget, method=method, arm=arm, repeat=repeat,
                       fits=budget, winner=winner, temperature=temperature,
                       validation_brier=validation_brier, **times)
            records.append(row)
            pending.append((row, weights, check))
    # All decisions are fixed before evaluating any test logits or test Brier.
    for row, weights, check in pending:
        logits = features[test_rows] @ weights
        row["test_brier"] = softmax_brier(logits, labels[test_rows], row["temperature"])
        row["top1_accuracy"] = float(np.mean(logits.argmax(axis=1) == labels[test_rows]))
        if not (math.isclose(row["test_brier"], float(check["brier"]), rel_tol=0, abs_tol=1e-8)
                and math.isclose(row["top1_accuracy"], float(check["accuracy"]),
                                 rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f"Test metrics differ from archived calibration: {row}")
    summary = []
    for budget, method, arm in jobs:
        group = [row for row in records if (row["budget"], row["method"], row["arm"])
                 == (budget, method, arm)]
        summary.append(dict(unit=unit, budget=budget, method=method, arm=arm, repeats=repeats,
                            fits=budget, winner=group[0]["winner"], temperature=group[0]["temperature"],
                            test_brier=group[0]["test_brier"], top1_accuracy=group[0]["top1_accuracy"],
                            **{key: median(row[key] for row in group) for key in STAGES}))
    print(f"PASS {unit}: {len(records)} independent runs match frozen winners and test metrics", flush=True)
    return dict(unit=unit, input_seconds=input_seconds, original_t1_gate=gate,
                repetitions=records, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--units", nargs="+", default=["resnet50/clipart"], metavar="ENCODER/DOMAIN")
    parser.add_argument("--methods", nargs="+", default=["target_a", "second_moment_mmd"])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    methods = {"target_a", "second_moment_mmd", *(f"random:{i}" for i in range(20))}
    if args.repeats < 1 or len(set(args.methods)) != len(args.methods) or not set(args.methods) <= methods:
        parser.error("Provide positive repeats and unique archived screening methods")
    if args.output.exists():
        parser.error("Output already exists")
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        if any(pool["num_threads"] != 1 for pool in pools):
            raise RuntimeError("Could not enforce a single numerical-library thread")
        units = [run_unit(args.cache_root / unit, unit, args.methods, args.repeats) for unit in args.units]
    payload = dict(status="complete", projection_seed=20260911, ridge_lambda=1.0,
                   scope="resident projected features and archived rankings through validation choice",
                   excluded=["ranking", "feature extraction", "common disk input", "test evaluation"],
                   temperature_grid=dict(minimum=.01, maximum=100.0, points=len(TEMPERATURES)),
                   environment=dict(python=platform.python_version(), numpy=np.__version__,
                                    scipy=scipy.__version__, platform=platform.platform(), pools=pools),
                   units=units)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as sink:
        json.dump(payload, sink, indent=2)
        sink.write("\n")


if __name__ == "__main__":
    main()
