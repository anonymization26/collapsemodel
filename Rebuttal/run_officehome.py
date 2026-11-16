"""Run the frozen Office-Home supplemental protocol on cached features."""
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
from statistics import mean
import sys
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

cache, output = map(Path, sys.argv[1:])
output.mkdir(exist_ok=False)
domains = ("Art", "Clipart", "Product", "Real-World")
seed = 20261107
budget = 50

def load(split, domain):
    path = cache / split / f"{domain}.npz"
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
    with np.load(path, allow_pickle=False) as data:
        ids = data["ids"].tolist()
        assert len(set(ids)) == len(ids) == metadata["rows"]
        assert np.all(data["split"] == split)
        assert data["class_names"].tolist() == classes
        return ids, data["h"], data["y"]

def order(ids):
    return sorted(range(len(ids)), key=lambda i: (hashlib.sha256(
        f"officehome-v1:{ids[i]}".encode()).digest(), ids[i]))

def normalized(matrix):
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)

def project(matrix):
    return normalized(normalized(matrix) @ projection).astype(np.float64)

def save(name, value):
    (output / name).write_text(json.dumps(value, indent=2) + "\n")

with np.load(cache / "train/Art.npz", allow_pickle=False) as first:
    classes = first["class_names"].tolist()
    dim = first["h"].shape[1]
assert len(classes) == 65
signs = np.random.default_rng(seed).integers(0, 2, (dim, 128), dtype=np.int8)
projection = (2 * signs.astype(np.float32) - 1) / np.float32(np.sqrt(128))
blocks, scatters, membership, support, train_ids = {}, {}, {}, {}, {}
for domain in domains:
    ids, raw, labels = load("train", domain)
    train_ids[domain] = set(ids)
    selected = order(ids)[:1024]
    assert len(selected) == 1024
    for block, indices in enumerate(np.array(selected).reshape(4, 256)):
        name = f"{domain}:{block}"
        h = project(raw[indices])
        blocks[name] = h, labels[indices]
        scatters[name] = h.T @ h
        membership[name] = [ids[i] for i in indices]
    ids, raw, labels = load("support_pool", domain)
    assert train_ids[domain].isdisjoint(ids)
    indices = order(ids)
    support[domain] = (ids, project(raw), labels, indices[::2], indices[1::2])

rankings, candidates, support_ids = {}, {}, {}
for domain, (ids, h, _, screen, val) in support.items():
    support_ids[domain] = {"screen": [ids[i] for i in screen], "validation": [ids[i] for i in val]}
    target = h[screen].T @ h[screen] / len(screen)
    combos = {"|".join(c): c for c in combinations(sorted(
        name for name in blocks if name.split(":")[0] != domain), 3)}
    assert len(combos) == 220
    candidates[domain] = combos
    scores = {method: {} for method in ("a_opt", "target_energy", "mmd")}
    for name, combo in combos.items():
        gram = sum(scatters[block] for block in combo)
        eigenvalues, vectors = np.linalg.eigh(gram)
        scores["a_opt"][name] = float(np.sum(np.sum(vectors * (target @ vectors), axis=0)
                                            / (1 + np.maximum(eigenvalues, 0))))
        moment = gram / 768
        scores["target_energy"][name] = -float(np.sum(target * moment))
        scores["mmd"][name] = float(np.sum((target - moment) ** 2))
    rankings[domain] = {method: sorted(values, key=lambda key: (values[key], key))
                        for method, values in scores.items()}
    for repeat in range(20):
        rankings[domain][f"random:{repeat}"] = np.random.default_rng(seed + repeat).permutation(
            sorted(combos)).tolist()
save("frozen_screen.json", dict(membership=membership, candidates=candidates,
                                support_ids=support_ids, rankings=rankings))
print("Frozen screen: 4 targets, 220 candidates per target, budget 50", flush=True)

def brier(model, h, y):
    probability = model.predict_proba(h)
    columns = np.zeros((len(y), len(classes)))
    columns[:, model.classes_] = probability
    columns[np.arange(len(y)), y] -= 1
    return float(np.mean(np.sum(columns ** 2, axis=1)))

warnings.simplefilter("error", ConvergenceWarning)
results, details = [], {}
for domain, (ids, h, labels, _, val) in support.items():
    orders = rankings[domain]
    shortlist = {method: ranking[:budget] for method, ranking in orders.items()}
    union = sorted(set().union(*shortlist.values()))
    models, validation = {}, {}
    for key in union:
        pieces = [blocks[block] for block in candidates[domain][key]]
        x, y = map(np.concatenate, zip(*pieces))
        model = LogisticRegression(C=1, solver="lbfgs", max_iter=1000).fit(x, y)
        models[key] = model
        validation[key] = brier(model, h[val], labels[val])
    winners = {method: min(prefix, key=lambda key: (validation[key], key))
               for method, prefix in shortlist.items()}
    test_ids, test_raw, test_y = load("test", domain)
    assert set(test_ids).isdisjoint(train_ids[domain] | set(ids))
    test_h = project(test_raw)
    losses = {key: brier(models[key], test_h, test_y) for key in set(winners.values())}
    by_method = {method: losses[key] for method, key in winners.items()}
    row = dict(domain=domain, **{m: by_method[m] for m in ("a_opt", "target_energy", "mmd")},
               random_mean=mean(by_method[f"random:{i}"] for i in range(20)))
    results.append(row)
    details[domain] = dict(validation=validation, winners=winners, test_brier=by_method)
    print(f"{domain}: fitted {len(union)} candidates; test rows {len(test_y)}; {row}", flush=True)
results.append(dict(domain="overall", **{key: mean(row[key] for row in results)
                                        for key in results[0] if key != "domain"}))
for row in results:
    for method in ("target_energy", "mmd", "random_mean"):
        row[f"a_opt_minus_{method}"] = row["a_opt"] - row[method]
with (output / "results.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)
save("details.json", details)
