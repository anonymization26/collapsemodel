#!/usr/bin/env python3
"""Run all leave-one-source-out screens after one shared spectral preprocessing pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import two_stage_classic_baselines as classic


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_methods(
    names: list[str],
    ranks: dict[str, float],
    similarity: np.ndarray,
    max_shortlist: int,
) -> tuple[dict[str, list[str]], dict[str, float]]:
    selections = {}
    runtimes = {}

    def select(method: str, function) -> None:
        started = time.perf_counter()
        selected = function()
        runtimes[method] = time.perf_counter() - started
        classic.validate_selection(selected, names, max_shortlist)
        selections[method] = selected

    select("rank_only", lambda: classic.rank_only(ranks, max_shortlist))
    select(
        "facility_subspace",
        lambda: classic.facility_location(similarity, names, max_shortlist),
    )
    select(
        "kcenter_subspace",
        lambda: classic.k_center(similarity, names, ranks, max_shortlist),
    )
    select(
        "dpp_subspace",
        lambda: classic.dpp_greedy(similarity, names, ranks, max_shortlist),
    )
    select(
        "pool_vendi_subspace",
        lambda: classic.pool_vendi_greedy(similarity, names, ranks, max_shortlist),
    )
    return selections, runtimes


def run(args: argparse.Namespace) -> None:
    import two_stage_natural_shortlist as natural

    if max(args.shortlist_sizes) >= len(natural.SOURCES):
        raise ValueError(
            "shortlist must be smaller than the leave-one-source candidate set"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_features = {}
    for source in natural.SOURCES:
        features, _ = natural.load_cache(natural.source_path(
            args.source_dir, args.encoder, source, args.cache_samples,
        ))
        all_features[source] = natural.sample_unlabeled(
            features,
            args.stage1_samples,
            natural.stable_seed(source, args.sample_seed),
        )

    all_names = sorted(all_features)
    excluded_sources = args.excluded_sources or all_names
    unknown_sources = sorted(set(excluded_sources) - set(all_names))
    if unknown_sources:
        raise ValueError(f"unknown excluded sources: {unknown_sources}")
    preprocessing_started = time.perf_counter()
    ranks, _, centroids, subspaces = classic.pool_statistics(
        all_features, args.top_k,
    )
    full_similarity = classic.similarity_matrix(
        all_features, all_names, "subspace", centroids, subspaces,
    )
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    max_shortlist = max(args.shortlist_sizes)
    script_hash = sha256_file(Path(__file__))
    natural_script_hash = sha256_file(Path(natural.__file__))
    completed = []

    for excluded_source in excluded_sources:
        names = [name for name in all_names if name != excluded_source]
        indices = [all_names.index(name) for name in names]
        similarity = full_similarity[np.ix_(indices, indices)]
        subset_ranks = {name: ranks[name] for name in names}
        sequences, runtimes = select_methods(
            names, subset_ranks, similarity, max_shortlist,
        )
        selections = {
            str(shortlist_size): {
                method: selected[:shortlist_size]
                for method, selected in sequences.items()
            }
            for shortlist_size in args.shortlist_sizes
        }
        fold_dir = args.out_dir / f"exclude_{excluded_source}" / args.encoder
        fold_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = fold_dir / f"{args.encoder}_screening_manifest.json"
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "script_sha256": script_hash,
            "natural_shortlist_script_sha256": natural_script_hash,
            "encoder": args.encoder,
            "source_dir": str(args.source_dir),
            "sources": names,
            "source_domains": {
                source: natural.SOURCE_DOMAINS[source] for source in names
            },
            "source_families": {
                source: natural.SOURCE_FAMILIES[source] for source in names
            },
            "excluded_sources": [excluded_source],
            "targets": natural.TARGETS,
            "cache_samples": args.cache_samples,
            "stage1_samples": args.stage1_samples,
            "sample_seed": args.sample_seed,
            "top_k": args.top_k,
            "shortlist_sizes": args.shortlist_sizes,
            "shared_summary_preprocessing_seconds": preprocessing_seconds,
            "selections": selections,
            "selection_seconds_to_max_shortlist": runtimes,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        natural.run_summarize(Namespace(
            manifest=manifest_path,
            adaptation_csv=[args.adaptation_csv],
            out_dir=fold_dir,
            n_random=args.n_random,
            random_seed=args.random_seed,
            tie_tolerance=args.tie_tolerance,
        ))
        completed.append(excluded_source)
        print(
            f"SOURCE_DONE={excluded_source} "
            f"COMPLETED={len(completed)}/{len(excluded_sources)}",
            flush=True,
        )

    batch_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": script_hash,
        "natural_shortlist_script_sha256": natural_script_hash,
        "adaptation_csv": str(args.adaptation_csv),
        "adaptation_csv_sha256": sha256_file(args.adaptation_csv),
        "encoder": args.encoder,
        "source_count": len(all_names),
        "requested_exclusions": excluded_sources,
        "completed_exclusions": completed,
        "shortlist_sizes": args.shortlist_sizes,
        "shared_summary_preprocessing_seconds": preprocessing_seconds,
    }
    (args.out_dir / f"{args.encoder}_batch_manifest.json").write_text(
        json.dumps(batch_manifest, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--adaptation-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--cache-samples", type=int, default=5000)
    parser.add_argument("--stage1-samples", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=20260905)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--shortlist-sizes", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--excluded-sources", nargs="+")
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=20260905)
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
