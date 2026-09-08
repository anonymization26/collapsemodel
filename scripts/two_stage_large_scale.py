#!/usr/bin/env python3
"""Large-scale controlled-overlap experiment across pools, budgets, and seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import two_stage_overlap_experiment as core


SOURCES = [
    "bloodmnist", "breastmnist", "cifar10", "cifar100", "cifar100_coarse",
    "dermamnist", "fashion_mnist", "mnist", "octmnist", "organamnist",
    "organcmnist", "organsmnist", "pathmnist", "pneumoniamnist",
    "rendered_sst2", "retinamnist", "semeion", "stl10", "svhn",
    "tiny_imagenet", "tissuemnist", "usps",
]
TARGETS = [
    "beans", "dtd", "eurosat", "fgvc_aircraft", "flowers102", "food101",
    "gtsrb", "oxford_pets",
]
OVERLAPS = [0.25, 0.50, 0.75, 1.00]
CONSTRUCTION_SEEDS = [20260831, 20260832, 20260833]
BUDGETS = [3, 5]


def make_collection(source, overlap, pool_size, seed, duplicate_sources):
    original_sources = core.SOURCES
    try:
        core.SOURCES = SOURCES
        return core.make_collection(source, overlap, pool_size, seed, duplicate_sources)
    finally:
        core.SOURCES = original_sources


def family_rank_greedy(features, family, k):
    scores = {name: core.reff_numpy(h) for name, h in features.items()}
    selected, seen = [], set()
    for name in sorted(features, key=lambda item: (-scores[item], item)):
        if family[name] in seen:
            continue
        selected.append(name)
        seen.add(family[name])
        if len(selected) == k:
            break
    return selected


def run_screen(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = {name: core.load_feature(args.feature_dir, name) for name in SOURCES}
    full_reff = {name: core.reff_numpy(value[0]) for name, value in source.items()}
    duplicate_sources = sorted(full_reff, key=full_reff.get, reverse=True)[:args.n_duplicate_sources]
    manifest = {
        "sources": SOURCES,
        "targets": TARGETS,
        "construction_seeds": CONSTRUCTION_SEEDS,
        "overlaps": OVERLAPS,
        "budgets": BUDGETS,
        "pool_size": args.pool_size,
        "top_k": args.top_k,
        "duplicate_sources": duplicate_sources,
        "configs": {},
    }
    rows = []
    for construction_seed in CONSTRUCTION_SEEDS:
        for overlap in OVERLAPS:
            pools, family = make_collection(
                source, overlap, args.pool_size, construction_seed, duplicate_sources,
            )
            features = {name: value[0] for name, value in pools.items()}
            for budget in BUDGETS:
                key = f"seed{construction_seed}_overlap{overlap:.2f}_k{budget}"
                methods = {
                    "collapse": core.collapse_greedy(features, budget, args.top_k),
                    "r_sum": core.rank_greedy(features, budget),
                    "centroid_ff": core.centroid_ff(features, budget),
                    "family_rank": family_rank_greedy(features, family, budget),
                }
                rng = np.random.default_rng(construction_seed + budget * 1000 + int(overlap * 100))
                for index in range(args.n_random):
                    methods[f"random_{index}"] = sorted(
                        rng.choice(list(features), budget, replace=False).tolist()
                    )
                config = {
                    "construction_seed": construction_seed,
                    "overlap": overlap,
                    "budget": budget,
                    "family": family,
                    "methods": {},
                }
                reference = core.selection_metrics(methods["family_rank"], features, family)["merged_reff"]
                for method, selected in methods.items():
                    metrics = core.selection_metrics(selected, features, family)
                    metrics["rank_gap_vs_family"] = float(reference - metrics["merged_reff"])
                    config["methods"][method] = metrics
                    rows.append({
                        "config": key,
                        "construction_seed": construction_seed,
                        "overlap": overlap,
                        "budget": budget,
                        "method": method,
                        "selected": "|".join(selected),
                        **{name: value for name, value in metrics.items() if name != "selected"},
                    })
                manifest["configs"][key] = config
                print(
                    key,
                    f"collapse_dup={config['methods']['collapse']['duplicate_count']}",
                    f"rank_dup={config['methods']['r_sum']['duplicate_count']}",
                    flush=True,
                )
    (args.out_dir / "large_screening_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.out_dir / "large_screening_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run_adapt(args):
    manifest = json.loads((args.out_dir / "large_screening_manifest.json").read_text())
    source = {name: core.load_feature(args.feature_dir, name) for name in SOURCES}
    targets = {name: core.load_feature(args.feature_dir, name) for name in TARGETS}
    duplicate_sources = manifest["duplicate_sources"]
    device = torch.device(args.device)
    rows = []
    for key, config in manifest["configs"].items():
        methods = config["methods"]
        collapse_selected = methods["collapse"]["selected"]
        rank_selected = methods["r_sum"]["selected"]
        if collapse_selected == rank_selected and not args.run_agreements:
            print(f"skip {key}: collapse and r_sum agree", flush=True)
            continue
        pools, _ = make_collection(
            source, float(config["overlap"]), args.pool_size,
            int(config["construction_seed"]), duplicate_sources,
        )
        method_names = ["collapse", "r_sum", "centroid_ff", "family_rank"]
        method_names += sorted(name for name in methods if name.startswith("random_"))
        for method in method_names:
            selected = methods[method]["selected"]
            for adapter_seed in range(args.n_adapter_seeds):
                current = core.adapt_one(
                    selected, pools, targets, device, adapter_seed, args.steps,
                )
                for row in current:
                    row.update({
                        "config": key,
                        "construction_seed": config["construction_seed"],
                        "overlap": config["overlap"],
                        "budget": config["budget"],
                        "method": method,
                        "selected": "|".join(selected),
                    })
                rows.extend(current)
                print(key, method, adapter_seed, np.mean([row["accuracy"] for row in current]), flush=True)
    if not rows:
        print("No configurations selected for adaptation.")
        return
    with (args.out_dir / "large_adaptation_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["screen", "adapt"])
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="npu:7")
    parser.add_argument("--pool-size", type=int, default=250)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-duplicate-sources", type=int, default=6)
    parser.add_argument("--n-random", type=int, default=5)
    parser.add_argument("--n-adapter-seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--run-agreements", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.phase == "screen":
        run_screen(arguments)
    else:
        run_adapt(arguments)
