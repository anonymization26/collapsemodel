#!/usr/bin/env python3
"""Dataset-endpoint dependence analysis for Collapse Model pair results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def metrics(df: pd.DataFrame) -> dict[str, float]:
    y = df["r_merged_true"].to_numpy(float)
    p = df["r_merged_pred"].to_numpy(float)
    pearson = pearsonr(p, y).statistic
    spearman = spearmanr(p, y).statistic
    log_ratio = np.log(p / y)
    calibration = float(np.exp(-log_ratio.mean()))
    p_cal = calibration * p
    ss_res = float(np.square(y - p_cal).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    return {
        "n_pairs": int(len(df)),
        "pearson": float(pearson),
        "spearman": float(spearman),
        "calibration_c": calibration,
        "calibrated_r2": 1.0 - ss_res / ss_tot,
        "mean_log_bias": float(log_ratio.mean()),
    }


def expanded_endpoint_sample(
    df: pd.DataFrame, identities: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame | None:
    sampled = rng.choice(identities, size=len(identities), replace=True)
    names, counts = np.unique(sampled, return_counts=True)
    multiplicity = dict(zip(names.tolist(), counts.tolist()))
    weights = [
        multiplicity.get(a, 0) * multiplicity.get(b, 0)
        for a, b in zip(df["ds_A"], df["ds_B"])
    ]
    keep = np.flatnonzero(np.asarray(weights) > 0)
    if len(keep) < 4:
        return None
    repeated = np.repeat(keep, np.asarray(weights, dtype=int)[keep])
    out = df.iloc[repeated].reset_index(drop=True)
    if out["r_merged_true"].nunique() < 2 or out["r_merged_pred"].nunique() < 2:
        return None
    return out


def percentile(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, float)
    lo, hi = np.quantile(a, [0.025, 0.975])
    return {
        "mean": float(a.mean()),
        "se": float(a.std(ddof=1)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_csv", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260803)
    args = ap.parse_args()

    df = pd.read_csv(args.pairs_csv)
    identities = np.array(sorted(set(df["ds_A"]) | set(df["ds_B"])), dtype=object)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    full = metrics(df)
    rng = np.random.default_rng(args.seed)
    boot_rows = []
    while len(boot_rows) < args.n_boot:
        sample = expanded_endpoint_sample(df, identities, rng)
        if sample is not None:
            boot_rows.append(metrics(sample))
    boot = pd.DataFrame(boot_rows)

    lodo_rows = []
    for identity in identities:
        held = df[(df["ds_A"] != identity) & (df["ds_B"] != identity)]
        row = {"held_out_dataset": identity, **metrics(held)}
        lodo_rows.append(row)
    lodo = pd.DataFrame(lodo_rows)

    domain = {
        "bloodmnist": "medical", "dermamnist": "medical", "pathmnist": "medical",
        "comp": "text", "rec": "text", "sci": "text", "talk": "text",
    }
    domain_rows = []
    for d in ["natural", "medical", "text"]:
        members = {x for x in identities if domain.get(x, "natural") == d}
        held = df[~df["ds_A"].isin(members) & ~df["ds_B"].isin(members)]
        if len(held) >= 4:
            domain_rows.append({"held_out_domain": d, **metrics(held)})
    domain_df = pd.DataFrame(domain_rows)

    summary = {
        "method": "endpoint (dataset-identity) cluster bootstrap; resampled identities induce edge multiplicities m_A*m_B",
        "seed": args.seed,
        "n_boot": args.n_boot,
        "n_unique_dataset_identities": int(len(identities)),
        "full": full,
        "cluster_bootstrap": {
            col: percentile(boot[col].tolist())
            for col in ["pearson", "spearman", "calibration_c", "calibrated_r2"]
        },
        "lodo_range": {
            col: {"min": float(lodo[col].min()), "median": float(lodo[col].median()), "max": float(lodo[col].max())}
            for col in ["pearson", "spearman", "calibration_c", "calibrated_r2"]
        },
    }

    boot.to_csv(args.out_dir / "dataset_cluster_bootstrap.csv", index=False)
    lodo.to_csv(args.out_dir / "leave_one_dataset_out.csv", index=False)
    domain_df.to_csv(args.out_dir / "leave_one_domain_out.csv", index=False)
    (args.out_dir / "dependence_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
