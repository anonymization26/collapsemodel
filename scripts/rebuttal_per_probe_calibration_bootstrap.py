#!/usr/bin/env python3
"""Endpoint-cluster bootstrap for paper-consistent per-probe calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(d: pd.DataFrame, w: np.ndarray) -> tuple[float, float]:
    pred_cal = np.empty(len(d))
    coeffs = []
    for probe, idx in d.groupby("probe").indices.items():
        idx = np.asarray(idx)
        wp = w[idx]
        if wp.sum() == 0:
            continue
        c = np.exp(-np.average(np.log(d.r_merged_pred.to_numpy()[idx] / d.r_merged_true.to_numpy()[idx]), weights=wp))
        pred_cal[idx] = d.r_merged_pred.to_numpy()[idx] * c
        coeffs.append(c)
    active = w > 0
    y = d.r_merged_true.to_numpy()[active]
    yh = pred_cal[active]
    wa = w[active]
    ybar = np.average(y, weights=wa)
    r2 = 1 - np.sum(wa * (y - yh) ** 2) / np.sum(wa * (y - ybar) ** 2)
    return float(r2), float(np.mean(coeffs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    d = pd.read_csv(args.pairs).reset_index(drop=True)
    ids = np.asarray(sorted(set(d.ds_A) | set(d.ds_B)))
    pos = {x: i for i, x in enumerate(ids)}
    ia = np.asarray([pos[x] for x in d.ds_A])
    ib = np.asarray([pos[x] for x in d.ds_B])
    full_r2, full_mean_c = metrics(d, np.ones(len(d)))
    rng = np.random.default_rng(20260803)
    r2s, cs = np.empty(args.n_boot), np.empty(args.n_boot)
    for b in range(args.n_boot):
        counts = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)))
        weights = counts[ia] * counts[ib]
        r2s[b], cs[b] = metrics(d, weights)
    out = {
        "method": "dataset-identity endpoint cluster bootstrap with per-probe calibration",
        "n_pairs": len(d), "n_dataset_identities": len(ids), "n_boot": args.n_boot,
        "full_per_probe_calibrated_r2": full_r2,
        "bootstrap_r2_mean": float(r2s.mean()),
        "bootstrap_r2_95ci": [float(np.quantile(r2s, .025)), float(np.quantile(r2s, .975))],
        "full_mean_probe_coefficient": full_mean_c,
        "bootstrap_mean_coefficient_95ci": [float(np.quantile(cs, .025)), float(np.quantile(cs, .975))],
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
