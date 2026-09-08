#!/usr/bin/env python3
"""Merge four new vision-pair tables with the retained SBERT table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resnet", type=Path, required=True)
    ap.add_argument("--vit", type=Path, required=True)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--dino", type=Path, required=True)
    ap.add_argument("--sbert", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    vision = pd.concat([
        pd.read_csv(args.resnet).query("probe == 'resnet50'"),
        pd.read_csv(args.vit), pd.read_csv(args.clip), pd.read_csv(args.dino),
    ], ignore_index=True, sort=False)
    sbert = pd.read_csv(args.sbert)
    sbert["domain_A"] = "text"
    sbert["domain_B"] = "text"
    sbert["pair_type"] = "text"
    for col in vision.columns:
        if col not in sbert:
            sbert[col] = np.nan
    merged = pd.concat([vision, sbert[vision.columns]], ignore_index=True)
    if len(vision) != 264 or len(sbert) != 171 or len(merged) != 435:
        raise RuntimeError(f"unexpected counts: vision={len(vision)}, sbert={len(sbert)}, total={len(merged)}")
    if merged[["probe", "ds_A", "ds_B"]].duplicated().any():
        raise RuntimeError("duplicate probe/pair identities")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(merged.groupby("probe").size())
    print(f"total={len(merged)}, datasets={len(set(merged.ds_A) | set(merged.ds_B))}")


if __name__ == "__main__":
    main()
