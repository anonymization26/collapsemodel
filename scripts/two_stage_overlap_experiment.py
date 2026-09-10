#!/usr/bin/env python3
"""Controlled-overlap test for two-stage data-pool pre-screening.

Phase ``screen`` builds candidate collections with controlled sample overlap,
compares summary selectors with a full-feature oracle, and records only the
information needed to reproduce each collection. Phase ``adapt`` consumes the
screening manifest and trains fixed-budget adapters for configurations where
Collapse and rank-only selection disagree.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch_npu  # noqa: F401

from two_stage_classic_baselines import (
    collapse_predict,
    effective_rank as reff_numpy,
    effective_rank_from_singular_values,
    split_numerical_singular_values,
    stable_singular_values_and_vh,
)


SOURCES = [
    "cifar10", "cifar100", "fashion_mnist", "mnist", "stl10", "svhn",
    "bloodmnist", "dermamnist", "pathmnist",
]
TARGETS = ["dtd", "eurosat", "flowers102", "oxford_pets", "food101"]
OVERLAPS = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_feature(root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(root / f"{name}_resnet50_N1000.npz")
    h = z["H"].astype(np.float32)
    h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-8)
    y = z["y"].astype(np.int64)
    _, y = np.unique(y, return_inverse=True)
    return h, y.astype(np.int64)


def summary(h: np.ndarray, k: int) -> dict[str, object]:
    singular, vh = stable_singular_values_and_vh(h)
    singular, _ = split_numerical_singular_values(singular, h.shape)
    return {
        "reff": effective_rank_from_singular_values(singular, h.shape),
        "nuclear": float(singular.sum()),
        "vh": vh[: min(k, len(singular))].astype(np.float32),
    }


def make_collection(
    source: dict[str, tuple[np.ndarray, np.ndarray]], overlap: float,
    pool_size: int, seed: int, duplicate_sources: list[str],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, str]]:
    pools: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    family: dict[str, str] = {}
    base_indices: dict[str, np.ndarray] = {}
    complement_indices: dict[str, np.ndarray] = {}

    for offset, name in enumerate(SOURCES):
        h, y = source[name]
        if len(h) < 2 * pool_size:
            raise ValueError(f"{name} has {len(h)} rows; need at least {2 * pool_size}")
        rng = np.random.default_rng(seed + offset)
        perm = rng.permutation(len(h))
        base = perm[:pool_size]
        complement = perm[pool_size:2 * pool_size]
        base_indices[name] = base
        complement_indices[name] = complement
        pools[name] = (h[base], y[base])
        family[name] = name

    shared = int(round(pool_size * overlap))
    for offset, name in enumerate(duplicate_sources):
        h, y = source[name]
        rng = np.random.default_rng(seed + 10_000 + offset)
        shared_idx = rng.choice(base_indices[name], shared, replace=False) if shared else np.array([], dtype=int)
        fresh_count = pool_size - shared
        fresh_idx = rng.choice(complement_indices[name], fresh_count, replace=False) if fresh_count else np.array([], dtype=int)
        idx = np.concatenate([shared_idx, fresh_idx])
        rng.shuffle(idx)
        alias = f"{name}__overlap_{int(overlap * 100):03d}"
        pools[alias] = (h[idx], y[idx])
        family[alias] = name
    return pools, family


def collapse_greedy(features: dict[str, np.ndarray], k: int, top_k: int) -> list[str]:
    stats = {name: summary(h, top_k) for name, h in features.items()}
    selected = [max(features, key=lambda name: stats[name]["reff"])]
    current = features[selected[0]]
    while len(selected) < k:
        cur = summary(current, top_k)
        best_name, best_score = None, -np.inf
        for name in features:
            if name in selected:
                continue
            candidate = stats[name]
            cosines = np.linalg.svd(cur["vh"] @ candidate["vh"].T, compute_uv=False)
            alpha = float(np.mean(np.clip(cosines, 0, 1) ** 2))
            gamma = float(candidate["nuclear"] / cur["nuclear"])
            score = collapse_predict(float(cur["reff"]), float(candidate["reff"]), gamma, alpha)
            if score > best_score:
                best_name, best_score = name, score
        assert best_name is not None
        selected.append(best_name)
        current = np.concatenate([current, features[best_name]], axis=0)
    return selected


def rank_greedy(features: dict[str, np.ndarray], k: int) -> list[str]:
    scores = {name: reff_numpy(h) for name, h in features.items()}
    return sorted(features, key=lambda name: (-scores[name], name))[:k]


def centroid_ff(features: dict[str, np.ndarray], k: int) -> list[str]:
    centroids = {}
    scores = {}
    for name, h in features.items():
        c = h.mean(axis=0)
        centroids[name] = c / max(np.linalg.norm(c), 1e-8)
        scores[name] = reff_numpy(h)
    selected = [max(features, key=lambda name: scores[name])]
    while len(selected) < k:
        remaining = [name for name in features if name not in selected]
        selected.append(max(
            remaining,
            key=lambda name: min(1 - float(centroids[name] @ centroids[s]) for s in selected),
        ))
    return selected


def oracle_greedy(features: dict[str, np.ndarray], k: int) -> list[str]:
    selected = [max(features, key=lambda name: reff_numpy(features[name]))]
    current = features[selected[0]]
    while len(selected) < k:
        remaining = [name for name in features if name not in selected]
        best = max(remaining, key=lambda name: reff_numpy(np.concatenate([current, features[name]], axis=0)))
        selected.append(best)
        current = np.concatenate([current, features[best]], axis=0)
    return selected


def selection_metrics(
    selected: list[str], features: dict[str, np.ndarray], family: dict[str, str],
) -> dict[str, object]:
    merged = np.concatenate([features[name] for name in selected], axis=0)
    unique = len({family[name] for name in selected})
    return {
        "selected": selected,
        "merged_reff": reff_numpy(merged),
        "unique_families": unique,
        "duplicate_count": len(selected) - unique,
    }


def run_screen(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = {name: load_feature(args.feature_dir, name) for name in SOURCES}
    full_reff = {name: reff_numpy(value[0]) for name, value in source.items()}
    duplicate_sources = sorted(full_reff, key=full_reff.get, reverse=True)[:args.n_duplicate_sources]
    manifest: dict[str, object] = {
        "seed": args.seed,
        "pool_size": args.pool_size,
        "k": args.k,
        "top_k": args.top_k,
        "duplicate_sources": duplicate_sources,
        "overlaps": {},
    }
    rows = []
    for overlap in OVERLAPS:
        pools, family = make_collection(source, overlap, args.pool_size, args.seed, duplicate_sources)
        features = {name: value[0] for name, value in pools.items()}
        methods = {
            "collapse": collapse_greedy(features, args.k, args.top_k),
            "r_sum": rank_greedy(features, args.k),
            "centroid_ff": centroid_ff(features, args.k),
            "oracle": oracle_greedy(features, args.k),
        }
        rng = np.random.default_rng(args.seed + int(overlap * 1000))
        for idx in range(args.n_random):
            methods[f"random_{idx}"] = sorted(rng.choice(list(features), args.k, replace=False).tolist())
        overlap_key = f"{overlap:.2f}"
        manifest["overlaps"][overlap_key] = {"family": family, "methods": {}}
        oracle_value = selection_metrics(methods["oracle"], features, family)["merged_reff"]
        for method, selected in methods.items():
            metrics = selection_metrics(selected, features, family)
            metrics["regret_vs_oracle"] = float(oracle_value - metrics["merged_reff"])
            manifest["overlaps"][overlap_key]["methods"][method] = metrics
            rows.append({
                "overlap": overlap,
                "method": method,
                "selected": "|".join(selected),
                **{key: value for key, value in metrics.items() if key != "selected"},
            })
        print(overlap_key, methods["collapse"], methods["r_sum"], flush=True)
    (args.out_dir / "screening_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.out_dir / "screening_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


class Adapter(nn.Module):
    def __init__(self, d_in: int, d_out: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_out, bias=False), nn.LayerNorm(d_out), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def stratified_split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train, test = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        cut = max(1, int(0.8 * len(idx)))
        train.extend(idx[:cut]); test.extend(idx[cut:])
    return np.asarray(train), np.asarray(test)


def ridge_accuracy(z_train: torch.Tensor, y_train: torch.Tensor, z_test: torch.Tensor, y_test: torch.Tensor) -> float:
    classes = int(y_train.max().item()) + 1
    onehot = torch.nn.functional.one_hot(y_train, classes).float()
    eye = torch.eye(z_train.shape[1], device=z_train.device)
    weights = torch.linalg.solve(z_train.T @ z_train + 1e-2 * eye, z_train.T @ onehot)
    return float(((z_test @ weights).argmax(1) == y_test).float().mean().cpu())


def adapt_one(
    selected: list[str], pools: dict[str, tuple[np.ndarray, np.ndarray]],
    targets: dict[str, tuple[np.ndarray, np.ndarray]], device: torch.device,
    seed: int, steps: int,
) -> list[dict[str, object]]:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    d_in = next(iter(pools.values()))[0].shape[1]
    adapter = Adapter(d_in).to(device)
    heads = nn.ModuleDict({
        name.replace(".", "_"): nn.Linear(256, int(pools[name][1].max()) + 1).to(device)
        for name in selected
    })
    optimizer = torch.optim.AdamW(list(adapter.parameters()) + list(heads.parameters()), lr=2e-3, weight_decay=1e-4)
    tensors = {
        name: (torch.from_numpy(pools[name][0]).to(device), torch.from_numpy(pools[name][1]).to(device))
        for name in selected
    }
    rng = np.random.default_rng(seed)
    adapter.train()
    for step in range(steps):
        name = selected[step % len(selected)]
        x, y = tensors[name]
        idx = torch.from_numpy(rng.integers(0, len(x), size=128)).to(device)
        loss = torch.nn.functional.cross_entropy(heads[name.replace(".", "_")](adapter(x[idx])), y[idx])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    adapter.eval()
    rows = []
    with torch.no_grad():
        for target, (h, y) in targets.items():
            train, test = stratified_split(y, seed + 17)
            z = adapter(torch.from_numpy(h).to(device))
            train_t = torch.from_numpy(train).to(device); test_t = torch.from_numpy(test).to(device)
            accuracy = ridge_accuracy(
                z[train_t], torch.from_numpy(y[train]).to(device),
                z[test_t], torch.from_numpy(y[test]).to(device),
            )
            rows.append({"seed": seed, "target": target, "accuracy": accuracy})
    return rows


def run_adapt(args: argparse.Namespace) -> None:
    manifest = json.loads((args.out_dir / "screening_manifest.json").read_text())
    source = {name: load_feature(args.feature_dir, name) for name in SOURCES}
    targets = {name: load_feature(args.feature_dir, name) for name in TARGETS}
    duplicate_sources = manifest["duplicate_sources"]
    device = torch.device(args.device)
    rows = []
    for overlap_key, result in manifest["overlaps"].items():
        methods = result["methods"]
        if methods["collapse"]["selected"] == methods["r_sum"]["selected"]:
            print(f"skip overlap={overlap_key}: collapse and r_sum agree", flush=True)
            continue
        pools, _ = make_collection(source, float(overlap_key), args.pool_size, args.seed, duplicate_sources)
        method_names = ["collapse", "r_sum", "centroid_ff", "oracle"]
        method_names += sorted(name for name in methods if name.startswith("random_"))
        for method in method_names:
            selected = methods[method]["selected"]
            for seed in range(args.n_seeds):
                current = adapt_one(selected, pools, targets, device, seed, args.steps)
                for row in current:
                    row.update({"overlap": float(overlap_key), "method": method, "selected": "|".join(selected)})
                rows.extend(current)
                print(overlap_key, method, seed, np.mean([row["accuracy"] for row in current]), flush=True)
    if not rows:
        print("No Collapse/r_sum disagreement; no adaptation runs were needed.")
        return
    with (args.out_dir / "adaptation_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["screen", "adapt"])
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:7")
    parser.add_argument("--pool-size", type=int, default=500)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-duplicate-sources", type=int, default=3)
    parser.add_argument("--n-random", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.phase == "screen":
        run_screen(arguments)
    else:
        run_adapt(arguments)
