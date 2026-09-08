#!/usr/bin/env python3
"""Dependence-aware summary for the fixed-budget adaptation diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    keys = ["seed", "target"]
    collapse = df[df.method == "collapse"].set_index(keys).accuracy
    random_mean = (
        df[df.method.str.startswith("random_")]
        .groupby(keys).accuracy.mean()
    )
    paired = pd.DataFrame({"collapse": collapse, "random_mean": random_mean}).dropna()
    paired["difference"] = paired.collapse - paired.random_mean

    # Resample target identities; all seeds for a sampled target move together.
    targets = np.asarray(sorted(paired.index.get_level_values("target").unique()))
    by_target = paired.groupby(level="target").difference.mean()
    rng = np.random.default_rng(20260803)
    boot = np.empty(args.bootstrap)
    for i in range(args.bootstrap):
        sampled = rng.choice(targets, size=len(targets), replace=True)
        boot[i] = by_target.loc[sampled].mean()

    summary = {
        "n_targets": int(len(targets)),
        "n_seed_target_units": int(len(paired)),
        "collapse_mean_accuracy": float(paired.collapse.mean()),
        "random_subset_mean_accuracy": float(paired.random_mean.mean()),
        "mean_difference": float(paired.difference.mean()),
        "target_cluster_bootstrap_95ci": [
            float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
        ],
        "per_target_mean_difference": {
            str(k): float(v) for k, v in by_target.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    paired.reset_index().to_csv(args.out.with_suffix(".csv"), index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
