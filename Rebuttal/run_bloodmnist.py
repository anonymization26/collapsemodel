"""Evaluate the fixed BloodMNIST pool screening protocol."""
import hashlib
from itertools import combinations
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

cache, output = map(Path, sys.argv[1:])
output.mkdir(exist_ok=False)
seed, budget, n_classes = 20261107, 50, 8

def load(split):
    path = cache / f"{split}.npz"
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["feature_file_sha256"]
    with np.load(path, allow_pickle=False) as data:
        ids = data["sample_ids"].tolist()
        assert len(ids) == len(set(ids))
        return ids, data["H"]

def order(ids):
    return sorted(range(len(ids)), key=lambda i: (
        hashlib.sha256(f"bloodmnist-v1:{ids[i]}".encode()).digest(), ids[i]))

def normalize(h):
    return h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-8)

def project(h):
    return normalize(normalize(h) @ projection).astype(np.float64)

def save(name, data):
    (output / name).write_text(json.dumps(data, indent=2) + "\n")

source_ids, source = load("train")
target_ids, target = load("val")
assert set(source_ids).isdisjoint(target_ids)
signs = np.random.default_rng(seed).integers(0, 2, (source.shape[1], 128), dtype=np.int8)
projection = (2 * signs.astype(np.float32) - 1) / np.float32(np.sqrt(128))
chosen = order(source_ids)[:3840]
source = project(source[chosen])
block_order = sorted(range(len(chosen)), key=lambda i: (source[i, 0], source_ids[chosen[i]]))
blocks = np.array(block_order).reshape(15, 256)
target_order = order(target_ids)
screen, validate, holdout = [target_order[i::3] for i in range(3)]
target_screen = project(target[screen])
moment = target_screen.T @ target_screen / len(screen)
grams = [source[block].T @ source[block] for block in blocks]
candidates = {"|".join(f"{i:02d}" for i in combo): combo for combo in combinations(range(15), 3)}
assert len(candidates) == 455

def score(method, gram):
    if method == "a_opt":
        eigenvalues, vectors = np.linalg.eigh(gram)
        energy = np.sum(vectors * (moment @ vectors), axis=0)
        return float(np.sum(energy / (1 + np.maximum(eigenvalues, 0))))
    if method == "target_energy":
        return -float(np.sum(moment * gram / 768))
    return float(np.sum((moment - gram / 768) ** 2))

rankings, timing = {}, {}
for method in ("a_opt", "target_energy", "mmd"):
    start = perf_counter()
    rankings[method] = sorted(candidates, key=lambda key: (
        score(method, sum(grams[i] for i in candidates[key])), key))
    timing[method] = perf_counter() - start
for repeat in range(20):
    rankings[f"random:{repeat}"] = np.random.default_rng(seed + repeat).permutation(sorted(candidates)).tolist()
save("frozen_screen.json", dict(
    blocks=[[source_ids[chosen[i]] for i in block] for block in blocks],
    roles={role: [target_ids[i] for i in indices]
           for role, indices in zip(("screen", "validation", "holdout"), (screen, validate, holdout))},
    candidates=candidates, rankings=rankings))
print(f"Frozen 455 candidates and target roles: {len(screen)}/{len(validate)}/{len(holdout)}", flush=True)

def labels(split, indices):
    with np.load(cache / f"{split}.npz", allow_pickle=False) as data:
        y = data["y"].reshape(-1)[indices]
    assert np.all((y >= 0) & (y < n_classes))
    return y

def brier(model, h, y):
    probability = np.zeros((len(y), n_classes))
    probability[:, model.classes_] = model.predict_proba(h)
    probability[np.arange(len(y)), y] -= 1
    return float(np.mean(np.sum(probability ** 2, axis=1)))

source_y = labels("train", chosen)
val_y = labels("val", validate)
val_h = project(target[validate])
prefixes = {method: order[:budget] for method, order in rankings.items()}
union = sorted(set().union(*prefixes.values()))
warnings.simplefilter("error", ConvergenceWarning)
models, validation = {}, {}
for key in union:
    rows = blocks[list(candidates[key])].reshape(-1)
    model = LogisticRegression(C=1, solver="lbfgs", max_iter=1000).fit(source[rows], source_y[rows])
    models[key] = model
    validation[key] = brier(model, val_h, val_y)
winners = {method: min(prefix, key=lambda key: (validation[key], key))
           for method, prefix in prefixes.items()}
save("frozen_winners.json", winners)
held_h, held_y = project(target[holdout]), labels("val", holdout)
loss = {method: brier(models[key], held_h, held_y) for method, key in winners.items()}
summary = {key: loss[key] for key in ("a_opt", "target_energy", "mmd")}
summary["random_mean"] = mean(loss[f"random:{i}"] for i in range(20))
save("results.json", dict(test_brier=summary, per_repeat=loss, validation=validation,
                          screening_seconds=timing, fitted_candidates=len(union)))
print(f"Fitted {len(union)} candidates across all shortlists. Held-out Brier: {summary}", flush=True)
