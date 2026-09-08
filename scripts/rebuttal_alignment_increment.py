#!/usr/bin/env python3
"""Natural-pool incremental value of the alignment-aware predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def rho(x: pd.Series, y: pd.Series) -> float:
    return float(spearmanr(x, y).statistic)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_csv", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.pairs_csv).copy()
    df["r_sum"] = df["r_A"] + df["r_B"]
    df["r_max"] = df[["r_A", "r_B"]].max(axis=1)
    truth = df["r_merged_true"]
    overall = {
        "n_pairs": int(len(df)),
        "rho_collapse": rho(df["r_merged_pred"], truth),
        "rho_r_sum": rho(df["r_sum"], truth),
        "rho_r_max": rho(df["r_max"], truth),
    }

    quantiles = []
    df["alignment_quartile"] = pd.qcut(df["alpha"], 4, labels=False, duplicates="drop")
    for q, g in df.groupby("alignment_quartile"):
        quantiles.append({
            "alignment_quartile": int(q), "n": int(len(g)),
            "alpha_min": float(g.alpha.min()), "alpha_max": float(g.alpha.max()),
            "rho_collapse": rho(g.r_merged_pred, g.r_merged_true),
            "rho_r_sum": rho(g.r_sum, g.r_merged_true),
        })

    decisions = []
    for probe, g in df.groupby("probe"):
        identities = sorted(set(g.ds_A) | set(g.ds_B))
        for anchor in identities:
            candidates = g[(g.ds_A == anchor) | (g.ds_B == anchor)].copy()
            candidates["candidate"] = np.where(
                candidates.ds_A == anchor, candidates.ds_B, candidates.ds_A
            )
            if len(candidates) < 2:
                continue
            oracle_i = candidates.r_merged_true.idxmax()
            selections = {
                "collapse": candidates.r_merged_pred.idxmax(),
                "r_sum": candidates.r_sum.idxmax(),
                "r_max": candidates.r_max.idxmax(),
            }
            oracle_value = float(candidates.loc[oracle_i, "r_merged_true"])
            row = {
                "probe": probe, "anchor": anchor,
                "n_candidates": int(len(candidates)),
                "oracle_candidate": candidates.loc[oracle_i, "candidate"],
                "oracle_true_reff": oracle_value,
            }
            for method, idx in selections.items():
                value = float(candidates.loc[idx, "r_merged_true"])
                row[f"{method}_candidate"] = candidates.loc[idx, "candidate"]
                row[f"{method}_true_reff"] = value
                row[f"{method}_regret_pct"] = 100.0 * (oracle_value - value) / oracle_value
                row[f"{method}_oracle_hit"] = bool(idx == oracle_i)
            row["collapse_differs_from_r_sum"] = selections["collapse"] != selections["r_sum"]
            row["collapse_minus_r_sum_true_reff"] = (
                row["collapse_true_reff"] - row["r_sum_true_reff"]
            )
            decisions.append(row)
    decisions_df = pd.DataFrame(decisions)

    diff = decisions_df[decisions_df.collapse_differs_from_r_sum]
    decision_summary = {
        "n_anchor_tasks": int(len(decisions_df)),
        "collapse_oracle_hits": int(decisions_df.collapse_oracle_hit.sum()),
        "r_sum_oracle_hits": int(decisions_df.r_sum_oracle_hit.sum()),
        "r_max_oracle_hits": int(decisions_df.r_max_oracle_hit.sum()),
        "mean_regret_pct_collapse": float(decisions_df.collapse_regret_pct.mean()),
        "mean_regret_pct_r_sum": float(decisions_df.r_sum_regret_pct.mean()),
        "mean_regret_pct_r_max": float(decisions_df.r_max_regret_pct.mean()),
        "n_changed_vs_r_sum": int(len(diff)),
        "changed_wins": int((diff.collapse_minus_r_sum_true_reff > 1e-9).sum()),
        "changed_ties": int((diff.collapse_minus_r_sum_true_reff.abs() <= 1e-9).sum()),
        "changed_losses": int((diff.collapse_minus_r_sum_true_reff < -1e-9).sum()),
        "mean_gain_when_changed": float(diff.collapse_minus_r_sum_true_reff.mean()) if len(diff) else 0.0,
    }
    summary = {"overall": overall, "by_alignment_quartile": quantiles, "anchor_selection": decision_summary}
    decisions_df.to_csv(args.out_dir / "anchor_selection_tasks.csv", index=False)
    pd.DataFrame(quantiles).to_csv(args.out_dir / "alignment_quartiles.csv", index=False)
    (args.out_dir / "alignment_increment_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
