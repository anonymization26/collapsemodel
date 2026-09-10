#!/usr/bin/env python3
"""Build an additional 435-pair Collapse validation set from cached features.

Vision: four encoders x C(12, 2) = 264 pairs.
Text: merge the existing 171-pair SBERT table after vision computation.
The output retains endpoint identities for cluster bootstrap and LODO.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

from two_stage_classic_baselines import (
    collapse_4s,
    is_superadditive,
    numerical_singular_tolerance,
    superadditivity_delta,
)


DATASETS = [
    "cifar10", "cifar100", "stl10", "svhn", "mnist", "fashion_mnist",
    "bloodmnist", "dermamnist", "pathmnist", "dtd", "eurosat",
    "tiny_imagenet",
]
ENCODERS = ["resnet50", "vit_b16", "clip_vit_l14", "dino_vit_s16"]
DOMAINS = {
    "cifar10": "natural", "cifar100": "natural", "stl10": "natural",
    "svhn": "digit", "mnist": "handwritten", "fashion_mnist": "handwritten",
    "bloodmnist": "medical", "dermamnist": "medical", "pathmnist": "medical",
    "dtd": "texture", "eurosat": "satellite", "tiny_imagenet": "natural",
}
FIELDS = [
    "probe", "ds_A", "ds_B", "domain_A", "domain_B", "pair_type",
    "r_A", "r_B", "r_dom", "r_sub", "r_merged_true", "r_merged_pred",
    "alpha", "gamma", "nuclear_A", "nuclear_B", "r_star", "rho_c",
    "ratio", "delta_r", "pred_error", "pred_error_pct", "is_superadditive",
]


def feature_path(root: Path, dataset: str, encoder: str) -> Path:
    return root / f"{dataset}_{encoder}_N1000.npz"


def load_feature(path: Path, device: torch.device) -> torch.Tensor:
    z = np.load(path)
    h = z["H"].astype(np.float32)
    h /= np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
    return torch.from_numpy(h).to(device)


def spectral_stats(h: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    _, s, vh = torch.linalg.svd(h, full_matrices=False)
    tolerance = numerical_singular_tolerance(float(s.max().cpu()), tuple(h.shape))
    retained = s > tolerance
    s = s[retained]
    vh = vh[retained]
    p = s / s.sum()
    r = torch.exp(-(p * torch.log(p)).sum())
    return float(r.cpu()), float(s.sum().cpu()), vh[:20]


def effective_rank(h: torch.Tensor) -> float:
    s = torch.linalg.svdvals(h)
    tolerance = numerical_singular_tolerance(float(s.max().cpu()), tuple(h.shape))
    s = s[s > tolerance]
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p)).sum()).cpu())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--encoders", nargs="+", choices=ENCODERS, default=ENCODERS)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    completed = set()
    if args.out.exists():
        with args.out.open() as f:
            completed = {(r["probe"], r["ds_A"], r["ds_B"]) for r in csv.DictReader(f)}
    write_header = not args.out.exists() or args.out.stat().st_size == 0

    with args.out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for encoder in args.encoders:
            missing = [d for d in DATASETS if not feature_path(args.feature_dir, d, encoder).exists()]
            if missing:
                raise FileNotFoundError(f"{encoder}: missing {missing}")
            features = {d: load_feature(feature_path(args.feature_dir, d, encoder), device) for d in DATASETS}
            stats = {d: spectral_stats(h) for d, h in features.items()}
            for a, b in itertools.combinations(DATASETS, 2):
                if (encoder, a, b) in completed:
                    continue
                r_a, nuc_a, vh_a = stats[a]
                r_b, nuc_b, vh_b = stats[b]
                cos = torch.linalg.svdvals(vh_a @ vh_b.T).clamp(0, 1)
                alpha = float((cos.square().mean()).cpu())
                gamma = nuc_b / nuc_a
                r_true = effective_rank(torch.cat([features[a], features[b]], dim=0))
                prediction = collapse_4s(r_a, r_b, gamma, alpha)
                r_pred = prediction.prediction
                r_dom = prediction.dominant_rank
                r_sub = prediction.subordinate_rank
                q = prediction.q
                rho_c = prediction.rho_c
                domain_a, domain_b = DOMAINS[a], DOMAINS[b]
                row = {
                    "probe": encoder, "ds_A": a, "ds_B": b,
                    "domain_A": domain_a, "domain_B": domain_b,
                    "pair_type": domain_a if domain_a == domain_b else f"{domain_a}x{domain_b}",
                    "r_A": r_a, "r_B": r_b, "r_dom": r_dom, "r_sub": r_sub,
                    "r_merged_true": r_true, "r_merged_pred": r_pred,
                    "alpha": alpha, "gamma": gamma, "nuclear_A": nuc_a, "nuclear_B": nuc_b,
                    "r_star": q, "rho_c": rho_c, "ratio": r_dom / r_sub,
                    "delta_r": superadditivity_delta(r_true, r_a, r_b, gamma),
                    "pred_error": abs(r_true - r_pred),
                    "pred_error_pct": 100 * abs(r_true - r_pred) / r_true,
                    "is_superadditive": is_superadditive(r_true, r_a, r_b, gamma),
                }
                writer.writerow(row); f.flush()
                print(encoder, a, b, r_true, r_pred, flush=True)


if __name__ == "__main__":
    main()
