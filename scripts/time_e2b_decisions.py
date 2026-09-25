#!/usr/bin/env python3
"""Measure standalone warm-feature decision paths without a model-result cache."""

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from threadpoolctl import threadpool_limits

from run_e2b_decision_challenge import score_method
from run_e2b_readout_projection import fit_readout, load_spec, model_scores, prediction_losses
from run_target_conditioned_e2b_shortlist import _projected_views, utc_now
from metrics.e2b_stage_bundle import load_stage_bundle
from metrics.target_conditioned_e2b import (
    canonical_json_sha256, load_config, parse_combination, read_json, sha256_file,
    stable_seed, write_json_atomic,
)


def execute(block_x, block_y, domains, selection_x, validation_x, validation_y,
            classes, method, readout, sizes, spec, seed):
    started = time.perf_counter()
    scores = score_method(block_x, domains, selection_x, 3, method)
    ranking = sorted(scores, key=lambda c: (scores[c], c))
    if method == "random":
        rng = np.random.default_rng(seed)
        ranking = [ranking[i] for i in rng.permutation(len(ranking))]
    score_seconds = time.perf_counter() - started
    budgets = [len(ranking)] if method == "exhaustive" else sizes
    losses, checkpoints = {}, []
    # Lazily cache block statistics so a small random prefix pays only for blocks it uses.
    if readout == "ridge":
        grams, rhs = {}, {}
    for i, combination in enumerate(ranking[:max(budgets)], 1):
        names = parse_combination(combination)
        if readout == "ridge":
            for name in names:
                if name not in grams:
                    x = np.asarray(block_x[name], dtype=np.float64)
                    grams[name] = x.T @ x
                    rhs[name] = x.T @ np.eye(classes)[block_y[name]]
            g = float(spec["ridge_regularization"]) * np.eye(validation_x.shape[1]) + sum(grams[n] for n in names)
            w = np.linalg.solve((g + g.T) / 2, sum(rhs[n] for n in names))
            active = np.ones(classes, dtype=bool)
        else:
            w, active, _ = fit_readout(np.concatenate([block_x[n] for n in names]),
                np.concatenate([block_y[n] for n in names]), classes, readout, spec)
        losses[combination] = prediction_losses(model_scores(validation_x, w, active), validation_y)
        if i in budgets:
            choices = {metric: min(losses, key=lambda c: (losses[c][metric], c))
                       for metric in spec["metrics"]}
            checkpoints.append({"shortlist_size": i, "score_seconds": score_seconds,
                                "decision_seconds": time.perf_counter() - started,
                                "selections": choices})
    return checkpoints, losses


def run(work, base, readout_protocol, protocol, encoder, output, readouts=None):
    config, spec, challenge = load_config(base), load_spec(readout_protocol), read_json(protocol)
    readouts = spec["readouts"] if readouts is None else readouts
    if not readouts or len(set(readouts)) != len(readouts) or not set(readouts) <= set(spec["readouts"]):
        raise ValueError("invalid timing readout subset")
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    config["features"]["projection"]["seed"] = challenge["projection_seed"]
    loaded = {}
    preparation = {}
    for stage in ("screen", "validate"):
        tick = time.perf_counter()
        data = load_stage_bundle(work / "stage_bundles" / encoder / stage, base, stage, encoder)
        x, idx, roles = _projected_views(data["features"], data["sample_ids"], data["samples"],
                                       data["candidate_ids"], config, encoder)
        preparation[stage] = time.perf_counter() - tick
        loaded[stage] = (data, x, idx, roles)
    records = []
    for domain in config["dataset"]["domains"]:
        sd, sx, si, sr = loaded["screen"]
        vd, vx, vi, vr = loaded["validate"]
        names = sorted(n for n,d in sd["candidate_domains"].items() if d != domain)
        bx = {n: vx[vi[n]] for n in names}
        by = {n: vd["labels"][vi[n]] for n in names}
        target_idx = vr[domain, "target_validation"]
        for readout in readouts:
            for repeat in range(challenge["timing"]["repeats"]):
                order = list(challenge["timing"]["methods"])
                np.random.default_rng(stable_seed(20260922, encoder, domain, readout, repeat)).shuffle(order)
                for method in order:
                    checkpoints, losses = execute(bx, by, sd["candidate_domains"],
                        sx[sr[domain, "target_selection"]], vx[target_idx], vd["labels"][target_idx],
                        int(vd["metadata"]["class_count"]), method, readout,
                        challenge["shortlist_sizes"], spec, stable_seed(20260911, encoder, domain, repeat))
                    identity = {"encoder": encoder, "target_domain": domain, "readout": readout,
                                "method": method, "repeat": repeat}
                    name = f"{domain}__{readout}__{repeat}__{method}.json"
                    write_json_atomic(output / name, {**identity, "checkpoints": checkpoints,
                                                       "validation_losses": losses})
                    records += [{**identity, **r} for r in checkpoints]
                print(f"timing {encoder} {domain} {readout} repeat={repeat} complete", flush=True)
    result = {"status": "complete", "records": records, "created_utc": utc_now(),
              "elapsed_seconds": time.perf_counter() - start,
              "input_preparation_seconds_all_domains": preparation,
              "preparation_accounting": "shared loading, integrity checks and projection recorded separately; not divided into fake per-task timings",
              "cold_feature_extraction_seconds": None,
              "cold_feature_status": "not-measured-do-not-claim-end-to-end-speedup",
              "timing_scope": "warm-resident-features-through-validation-choice; no test audit, disk output or raw feature extraction",
              "cross_method_model_cache": False,
              "readouts": readouts, "ridge_precision": "float64",
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "platform": platform.platform(), "pid": os.getpid()},
              "protocol_sha256": sha256_file(protocol), "base_config_sha256": sha256_file(base),
              "code_sha256": {p.name: sha256_file(p) for p in (
                  Path(__file__), Path(__file__).with_name("run_e2b_decision_challenge.py"),
                  Path(__file__).with_name("run_e2b_readout_projection.py"))}}
    result["result_id"] = canonical_json_sha256(result)
    write_json_atomic(output / "timing.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("work-root", "base-config", "readout-protocol", "protocol", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--readouts", nargs="+", choices=("ridge", "logistic"))
    a = parser.parse_args()
    with threadpool_limits(limits=1):
        result = run(a.work_root, a.base_config, a.readout_protocol, a.protocol, a.encoder, a.output_root, a.readouts)
    print(json.dumps({k: result[k] for k in ("status", "elapsed_seconds")}))
