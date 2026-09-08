#!/usr/bin/env python3
"""Fixed-budget feature-adapter diagnostic using selected source datasets.

The frozen ResNet-50 features are inputs, but the selected datasets directly
train a shared bottleneck adapter with dataset-specific supervised heads.
The adapter is then frozen and evaluated by ridge linear probes on disjoint
target datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch_npu  # noqa: F401


CANDIDATES = [
    "cifar10", "cifar100", "fashion_mnist", "mnist", "stl10", "svhn",
    "bloodmnist", "dermamnist", "pathmnist",
]
TARGETS = ["dtd", "eurosat", "flowers102", "oxford_pets", "food101"]


def path_for(root: Path, name: str) -> Path:
    return root / f"{name}_resnet50_N1000.npz"


def load(root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path_for(root, name))
    h = z["H"].astype(np.float32)
    y = z["y"].astype(np.int64)
    h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-8)
    _, y = np.unique(y, return_inverse=True)
    return h, y.astype(np.int64)


def spectrum(h: np.ndarray, device: torch.device) -> tuple[float, float, np.ndarray]:
    x = torch.from_numpy(h).to(device)
    _, s, vh = torch.linalg.svd(x, full_matrices=False)
    p = s / s.sum()
    reff = torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum())
    return float(reff.cpu()), float(s.sum().cpu()), vh[:20].float().cpu().numpy()


def collapse_predict(r_a: float, r_b: float, gamma: float, alpha: float) -> float:
    r_dom, r_sub = (r_a, r_b) if gamma <= 1 else (r_b, r_a)
    a = 1 + gamma * gamma
    d = math.sqrt(max((1 - gamma * gamma) ** 2 + 4 * gamma * gamma * alpha, 0.0))
    cp = math.sqrt((a + d) / 2)
    cm = math.sqrt(max((a - d) / 2, 1e-30))
    q = min(max(cp / (cp + cm), 1e-12), 1 - 1e-12)
    hb = -q * math.log(q) - (1 - q) * math.log(1 - q)
    return math.exp(hb + q * math.log(r_dom) + (1 - q) * math.log(r_sub))


def selections(
    features: dict[str, np.ndarray], device: torch.device, k: int
) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    stats = {}
    for name, h in features.items():
        r, nuc, vh = spectrum(h, device)
        stats[name] = {"reff": r, "nuclear": nuc, "vh": vh}
    ranked = sorted(features, key=lambda x: stats[x]["reff"], reverse=True)
    selected = [ranked[0]]
    current = features[selected[0]]
    while len(selected) < k:
        r_cur, nuc_cur, vh_cur = spectrum(current, device)
        best_name, best_score = None, -np.inf
        for name in features:
            if name in selected:
                continue
            cs = np.linalg.svd(vh_cur @ stats[name]["vh"].T, compute_uv=False)
            alpha = float(np.mean(np.clip(cs, 0, 1) ** 2))
            score = collapse_predict(r_cur, stats[name]["reff"], stats[name]["nuclear"] / nuc_cur, alpha)
            if score > best_score:
                best_name, best_score = name, score
        selected.append(best_name)
        current = np.concatenate([current, features[best_name]], axis=0)
    out = {"collapse": selected, "r_sum": ranked[:k]}
    rng = np.random.default_rng(20260803)
    for i in range(5):
        out[f"random_{i}"] = sorted(rng.choice(list(features), size=k, replace=False).tolist())
    serial_stats = {n: {"reff": s["reff"], "nuclear": s["nuclear"]} for n, s in stats.items()}
    return out, serial_stats


class Adapter(nn.Module):
    def __init__(self, d_in: int = 2048, d_out: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out, bias=False), nn.LayerNorm(d_out), nn.GELU()
        )

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
    w = torch.linalg.solve(z_train.T @ z_train + 1e-2 * eye, z_train.T @ onehot)
    pred = (z_test @ w).argmax(dim=1)
    return float((pred == y_test).float().mean().cpu())


def run_one(
    selected: list[str], source_data: dict[str, tuple[np.ndarray, np.ndarray]],
    target_data: dict[str, tuple[np.ndarray, np.ndarray]], device: torch.device,
    seed: int, steps: int,
) -> list[dict[str, object]]:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    adapter = Adapter().to(device)
    heads = nn.ModuleDict({
        name: nn.Linear(256, int(source_data[name][1].max()) + 1).to(device)
        for name in selected
    })
    opt = torch.optim.AdamW(list(adapter.parameters()) + list(heads.parameters()), lr=2e-3, weight_decay=1e-4)
    tensors = {
        name: (torch.from_numpy(source_data[name][0]).to(device), torch.from_numpy(source_data[name][1]).to(device))
        for name in selected
    }
    rng = np.random.default_rng(seed)
    adapter.train(); heads.train()
    for step in range(steps):
        name = selected[step % len(selected)]
        x, y = tensors[name]
        idx = torch.from_numpy(rng.integers(0, len(x), size=128)).to(device)
        loss = torch.nn.functional.cross_entropy(heads[name](adapter(x[idx])), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    adapter.eval()
    rows = []
    with torch.no_grad():
        for target, (h, y_np) in target_data.items():
            tr, te = stratified_split(y_np, seed + 17)
            x = torch.from_numpy(h).to(device)
            z = adapter(x)
            acc = ridge_accuracy(
                z[torch.from_numpy(tr).to(device)], torch.from_numpy(y_np[tr]).to(device),
                z[torch.from_numpy(te).to(device)], torch.from_numpy(y_np[te]).to(device),
            )
            rows.append({"seed": seed, "target": target, "accuracy": acc})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="npu:7")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    source = {n: load(args.feature_dir, n) for n in CANDIDATES}
    target = {n: load(args.feature_dir, n) for n in TARGETS}
    chosen, individual = selections({n: v[0] for n, v in source.items()}, device, args.k)
    (args.out_dir / "selections.json").write_text(json.dumps({"selections": chosen, "individual": individual}, indent=2) + "\n")
    all_rows = []
    for method, subset in chosen.items():
        for seed in [0, 1, 2]:
            rows = run_one(subset, source, target, device, seed, args.steps)
            for row in rows:
                row.update({"method": method, "selected": "|".join(subset)})
            all_rows.extend(rows)
            print(method, seed, np.mean([r["accuracy"] for r in rows]), flush=True)
    out_csv = args.out_dir / "adaptation_results.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    df = __import__("pandas").DataFrame(all_rows)
    summary = df.groupby("method").accuracy.agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    summary.to_csv(args.out_dir / "adaptation_summary.csv")
    print(summary)


if __name__ == "__main__":
    main()
