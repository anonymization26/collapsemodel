#!/usr/bin/env python3
"""Time matched Clipart logistic policies from resident projected features."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter
import warnings

import numpy as np
import scipy
from scipy.special import logsumexp
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_info, threadpool_limits

from run_pool_query import EXPLORATION, pool_design


METHODS = ("sequential25", "batch25", "mmd50", "full455")
BUDGETS = dict(zip(METHODS, (25, 25, 50, 455)))
LOGISTIC = {"C": 1.0, "solver": "lbfgs", "fit_intercept": False, "max_iter": 1000, "tol": 1e-6}
SCORE_ABSOLUTE_TOLERANCE = 1e-5


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def manifest(archive, stage, protocol_sha):
    meta = read_json(archive / stage / "manifest.json")
    assert meta["stage"] == stage and meta["status"] == "complete"
    assert meta["context"]["encoder"] == "resnet50"
    assert meta["context"]["projection_seed"] == 20260911
    assert meta["context"]["protocol_sha256"] == protocol_sha
    return meta


def checked_json(folder, meta, name):
    raw = (folder / name).read_bytes()
    assert digest(raw) == meta["files"][name], folder / name
    return json.loads(raw)


def aligned_indices(samples, domain, role, expected):
    indices = [i for i, sample in enumerate(samples)
               if sample["domain"] == domain and sample["role"] == role]
    assert len(indices) == expected
    return np.asarray(indices)


def load_inputs(ridge_dir, transport_dir):
    samples = read_json(ridge_dir / "temperature_inputs.json")["samples"]
    spec = read_json(transport_dir / "pilot.json")
    with np.load(ridge_dir / "verified_features.npz", allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float64)
    screening_features = np.asarray(
        np.load(transport_dir / "transport_features.npy", allow_pickle=False), dtype=np.float64
    )
    assert features.shape == (len(samples), 128)
    assert screening_features.shape == (len(spec["samples"]), 128)
    positions = {sample["sample_id"]: i for i, sample in enumerate(samples)}
    screening_positions = {sample["sample_id"]: i for i, sample in enumerate(spec["samples"])}
    assert len(positions) == len(samples) and len(screening_positions) == len(spec["samples"])
    blocks = {block["candidate_id"]: block["sample_ids"] for block in spec["blocks"]}
    assert len(blocks) == 15 and all(len(values) == len(set(values)) == 512 for values in blocks.values())
    if (ridge_dir / "pilot.json").exists():
        ridge_spec = read_json(ridge_dir / "pilot.json")
        original = {block["candidate_id"]: block["sample_ids"] for block in ridge_spec["blocks"]}
        assert original == blocks
    source_ids = sorted({sample for block in blocks.values() for sample in block})
    assert len(source_ids) == 7680
    for key in source_ids:
        left, right = samples[positions[key]], spec["samples"][screening_positions[key]]
        assert left["domain"] == right["domain"] and left["role"] == right["role"]
        assert left["domain"] != "clipart"
    source_error = float(np.max(np.abs(
        features[[positions[key] for key in source_ids]]
        - screening_features[[screening_positions[key] for key in source_ids]]
    )))
    # Both caches derive from the same frozen projection. Batch extraction order
    # can introduce small numerical differences.
    assert source_error <= SCORE_ABSOLUTE_TOLERANCE, source_error
    return (samples, features, screening_features, positions, screening_positions, blocks,
            aligned_indices(samples, "clipart", "target_validation", 1024),
            aligned_indices(samples, "clipart", "target_test", 2048),
            aligned_indices(spec["samples"], "clipart", "target_selection", 1536),
            source_error)


def fit_readout(features, labels):
    weights = np.zeros((features.shape[1], 100), dtype=np.float64)
    active = np.zeros(100, dtype=bool)
    observed = np.unique(labels)
    active[observed] = True
    if len(observed) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model = LogisticRegression(**LOGISTIC).fit(features, labels)
        if len(observed) == 2:
            weights[:, model.classes_[1]] = model.coef_[0]
        else:
            weights[:, model.classes_] = model.coef_.T
    return weights, active


def brier_score(features, labels, weights, active):
    scores = features @ weights
    scores[:, ~active] = -1e30
    probabilities = np.exp(scores - logsumexp(scores, axis=1, keepdims=True))
    actual = probabilities[np.arange(len(labels)), labels]
    return float(np.mean(np.sum(probabilities * probabilities, axis=1) - 2 * actual + 1))


def mmd_order(ids, blocks, target):
    grams = {name: values.T @ values for name, values in blocks.items()}
    target_moment = target.T @ target / len(target)
    scores = {}
    for candidate in ids:
        pooled = sum(grams[name] for name in candidate.split("|"))
        pooled = (pooled + pooled.T) / 2
        difference = pooled / 1536 - target_moment
        scores[candidate] = float(np.sum(difference * difference))
    return sorted(scores, key=lambda candidate: (scores[candidate], candidate))


def rank_predictions(design, queried, observed):
    matrix = design[queried]
    beta = np.linalg.lstsq(matrix, np.asarray(observed), rcond=None)[0]
    lower = np.linalg.cholesky(matrix.T @ matrix)
    leverage = np.linalg.norm(np.linalg.solve(lower, design.T), axis=0)
    scores = np.round(design @ beta - EXPLORATION * leverage, 12)
    scores[queried] = np.inf
    return np.argsort(scores, kind="stable").tolist()


def run_method(method, ids, inputs, labels, archived_mmd):
    (samples, features, screening_features, positions, screening_positions, blocks,
     validation, test, target, source_error) = inputs
    times = dict(preparation=0.0, design=0.0, screening=0.0, fit=0.0, validation=0.0, acquisition=0.0)
    start = perf_counter()
    tick = start
    indices = {name: np.array([positions[key] for key in members]) for name, members in blocks.items()}
    validation_features, validation_labels = features[validation], labels[validation]
    times["preparation"] = perf_counter() - tick
    queries, observed, weights = [], [], []
    mmd_gate = None

    def evaluate(candidate):
        tick = perf_counter()
        source = np.concatenate([indices[name] for name in candidate.split("|")])
        parameters = fit_readout(features[source], labels[source])
        times["fit"] += perf_counter() - tick
        tick = perf_counter()
        loss = brier_score(validation_features, validation_labels, *parameters)
        times["validation"] += perf_counter() - tick
        queries.append(candidate)
        observed.append(loss)
        weights.append(parameters)

    if method == "mmd50":
        tick = perf_counter()
        screening_blocks = {name: screening_features[[screening_positions[key] for key in members]]
                            for name, members in blocks.items()}
        ranking = mmd_order(ids, screening_blocks, screening_features[target])
        times["screening"] = perf_counter() - tick
        mmd_gate = {"exact_ranking": ranking == archived_mmd,
                    "top50_overlap": len(set(ranking[:50]) & set(archived_mmd[:50]))}
        assert mmd_gate["top50_overlap"] == 50, mmd_gate
        for candidate in ranking[:50]:
            evaluate(candidate)
    elif method == "full455":
        for candidate in ids:
            evaluate(candidate)
    else:
        tick = perf_counter()
        design, queried = pool_design(ids)
        times["design"] = perf_counter() - tick
        for index in queried:
            evaluate(ids[index])
        if method == "batch25":
            tick = perf_counter()
            additions = rank_predictions(design, queried, observed)[:10]
            times["acquisition"] = perf_counter() - tick
            for index in additions:
                evaluate(ids[index])
        else:
            while len(queries) < 25:
                tick = perf_counter()
                index = rank_predictions(design, queried, observed)[0]
                times["acquisition"] += perf_counter() - tick
                queried.append(index)
                evaluate(ids[index])
    winner_index = min(range(len(queries)), key=lambda i: (observed[i], queries[i]))
    times["total"] = perf_counter() - start
    assert len(queries) == len(set(queries)) == BUDGETS[method]
    return ({"method": method, "fit_count": len(queries), "seconds": times,
             "queries": queries, "validation_brier": observed, "winner": queries[winner_index],
             "mmd_alignment": mmd_gate}, weights[winner_index])


def expected_trace(method, ids, validation, archived_mmd):
    get = lambda candidate: validation[candidate]["brier_score"]
    if method == "full455":
        return ids
    if method == "mmd50":
        return archived_mmd[:50]
    design, queried = pool_design(ids)
    observed = [get(ids[index]) for index in queried]
    if method == "batch25":
        queried.extend(rank_predictions(design, queried, observed)[:10])
    else:
        while len(queried) < 25:
            index = rank_predictions(design, queried, observed)[0]
            queried.append(index)
            observed.append(get(ids[index]))
    return [ids[index] for index in queried]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ridge-dir", required=True, type=Path)
    parser.add_argument("--transport-dir", required=True, type=Path)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = args.protocol.read_bytes()
    protocol_sha = digest(protocol)
    assert json.loads(protocol)["logistic"] == LOGISTIC
    screen_meta = manifest(args.archive_dir, "screen", protocol_sha)
    val_meta = manifest(args.archive_dir, "validate", protocol_sha)
    assert val_meta["lineage"] == screen_meta["lineage"]
    assert val_meta["upstream"] == {"screen": screen_meta["artifact_id"]}
    rankings = checked_json(args.archive_dir / "screen", screen_meta, "rankings.json")
    archived_mmd = rankings["clipart"]["second_moment_mmd"]
    ids = sorted(archived_mmd)
    assert len(ids) == len(set(ids)) == 455
    inputs = load_inputs(args.ridge_dir, args.transport_dir)
    labels = np.asarray([int(sample["class_id"]) for sample in inputs[0]], dtype=np.int64)
    assert labels.min() >= 0 and labels.max() < 100
    args.output.mkdir(parents=True, exist_ok=False)
    runs, models = [], []
    with threadpool_limits(limits=1):
        libraries = threadpool_info()
        print(f"Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}, "
              f"scikit-learn {sklearn.__version__}, {platform.machine()}.", flush=True)
        for repeat in range(3):
            order = METHODS[repeat:] + METHODS[:repeat]
            for method in order:
                record, model = run_method(method, ids, inputs, labels, archived_mmd)
                record["repeat"] = repeat + 1
                models.append(model)
                runs.append(record)
                print(f"{repeat + 1} {method}: {record['seconds']['total']:.3f}s, "
                      f"{record['fit_count']} fits", flush=True)
    output = args.output
    metadata = {"encoder": "resnet50", "domain": "clipart", "seed": 20260911,
                "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                "python": platform.python_version(), "platform": platform.platform(), "threadpools": libraries,
                "timing_scope": "resident_projected_features_to_validation_winner",
                "logistic": LOGISTIC, "protocol_sha256": protocol_sha, "source_features_max_difference": inputs[-1]}
    write_json(output / "decisions.json", {"metadata": metadata, "runs": runs})
    decisions_sha = digest((output / "decisions.json").read_bytes())
    (output / "decisions.sha256").write_text(f"{decisions_sha}  decisions.json\n")

    validation = checked_json(args.archive_dir / "validate", val_meta, "clipart__logistic_losses.json")
    assert sorted(validation) == ids
    expected = {method: expected_trace(method, ids, validation, archived_mmd) for method in METHODS}
    maximum_error = 0.0
    for record in runs:
        method = record["method"]
        archive_winner = min(expected[method], key=lambda key: (validation[key]["brier_score"], key))
        errors = [abs(value - validation[key]["brier_score"])
                  for key, value in zip(record["queries"], record["validation_brier"])]
        record["validation_alignment"] = {
            "max_abs_brier_error": max(errors),
            "matches_reference_query_order": record["queries"] == expected[method],
            "matches_reference_query_set": set(record["queries"]) == set(expected[method]),
            "archive_winner": archive_winner, "winner_matches": record["winner"] == archive_winner,
        }
        maximum_error = max(maximum_error, max(errors))
    write_json(output / "validation_alignment.json", [r["validation_alignment"] for r in runs])
    assert maximum_error <= SCORE_ABSOLUTE_TOLERANCE, maximum_error

    # Test records are opened after all decisions have been written and hashed.
    assert digest((output / "decisions.json").read_bytes()) == decisions_sha
    test_meta = manifest(args.archive_dir, "test-audit", protocol_sha)
    assert test_meta["lineage"] == val_meta["lineage"]
    assert test_meta["upstream"] == {"screen": screen_meta["artifact_id"], "validate": val_meta["artifact_id"]}
    test = checked_json(args.archive_dir / "test-audit", test_meta, "clipart__logistic_losses.json")
    assert sorted(test) == ids
    test_rows = inputs[-3]
    with threadpool_limits(limits=1):
        for record, model in zip(runs, models):
            value = brier_score(inputs[1][test_rows], labels[test_rows], *model)
            archival = test[record["winner"]]["brier_score"]
            record["test_brier"] = value
            record["archived_winner_test_brier"] = archival
            record["test_brier_abs_error"] = abs(value - archival)
            assert record["test_brier_abs_error"] <= SCORE_ABSOLUTE_TOLERANCE
    summary = {}
    for method in METHODS:
        records = [r for r in runs if r["method"] == method]
        summary[method] = {
            "fit_count": BUDGETS[method],
            "median_seconds": {key: median(r["seconds"][key] for r in records)
                               for key in records[0]["seconds"]},
            "winner_ids": sorted({r["winner"] for r in records}),
            "test_brier": [r["test_brier"] for r in records],
            "archived_winner_agreement_count": sum(r["validation_alignment"]["winner_matches"] for r in records),
        }
    write_json(output / "timing.json", {"metadata": metadata, "decisions_sha256": decisions_sha,
                                        "summary": summary, "runs": runs})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
