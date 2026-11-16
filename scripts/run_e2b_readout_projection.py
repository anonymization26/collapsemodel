#!/usr/bin/env python3
"""Exploratory readout/projection audit with immutable role-separated stages."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from run_target_conditioned_e2b_shortlist import (
    ROOT, _projected_views, _score_combinations, git_revision, utc_now,
)
from metrics.e2b_stage_bundle import load_stage_bundle, TARGET_ROLE
from metrics.target_conditioned_e2b import (
    E2BArtifactError, canonical_json_sha256, load_config, parse_combination,
    read_json, sha256_file, stable_seed, write_json_atomic,
)


SOURCE_FILES = [
    "scripts/run_e2b_readout_projection.py",
    "scripts/run_target_conditioned_e2b_shortlist.py",
    "code/metrics/e2b_stage_bundle.py",
    "code/metrics/target_conditioned_e2b.py",
    "code/metrics/collapse_core.py",
]


def code_hashes():
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def load_spec(path):
    spec = read_json(path)
    if (spec.get("schema_version") != "e2b-readout-projection-v1"
            or spec.get("readouts") != ["ridge", "logistic"]
            or spec.get("metrics") != ["brier_score", "nll", "error_rate"]
            or spec.get("cross_method_result_cache") is not True
            or spec.get("logistic", {}).get("fit_intercept") is not False
            or spec.get("logistic", {}).get("solver") != "lbfgs"):
        raise E2BArtifactError("unexpected supplementary protocol")
    seeds = spec.get("projection_seeds", [])
    if not seeds or len(seeds) != len(set(seeds)):
        raise E2BArtifactError("projection seeds must be nonempty and unique")
    return spec


def check_artifact(directory, context, stage):
    artifact = read_json(directory / "manifest.json")
    core = {k: v for k, v in artifact.items() if k != "artifact_id"}
    if (artifact.get("artifact_id") != canonical_json_sha256(core)
            or artifact.get("context") != context or artifact.get("stage") != stage
            or artifact.get("status") != "complete"):
        raise E2BArtifactError("supplementary upstream identity mismatch")
    for name, digest in artifact["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise E2BArtifactError("invalid artifact file path")
        if sha256_file(directory / name) != digest:
            raise E2BArtifactError("supplementary upstream file hash mismatch")
    return artifact


def prediction_losses(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    logp = scores - logsumexp(scores, axis=1, keepdims=True)
    p = np.exp(logp)
    true_p = p[np.arange(len(truth)), truth]
    return {
        "brier_score": float(np.mean(np.sum(p * p, axis=1) - 2 * true_p + 1)),
        "nll": float(-np.mean(np.log(np.maximum(true_p, 1e-15)))),
        "error_rate": float(np.mean(scores.argmax(axis=1) != truth)),
    }


def fit_readout(x, y, classes, readout, spec):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    weights = np.zeros((x.shape[1], classes), dtype=np.float64)
    active = np.ones(classes, dtype=bool)
    iterations = 0
    if readout == "ridge":
        gram = x.T @ x + float(spec["ridge_regularization"]) * np.eye(x.shape[1])
        weights = np.linalg.solve((gram + gram.T) / 2, x.T @ np.eye(classes)[y])
    elif readout == "logistic":
        observed = np.unique(y)
        active[:] = False
        active[observed] = True
        if len(observed) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model = LogisticRegression(**spec["logistic"]).fit(x, y)
            if len(observed) == 2:
                # Binary sklearn stores one log-odds vector, not two logits.
                weights[:, model.classes_[1]] = model.coef_[0]
            else:
                weights[:, model.classes_] = model.coef_.T
            iterations = int(np.max(model.n_iter_))
    else:
        raise E2BArtifactError("unknown readout")
    if not np.isfinite(weights).all():
        raise E2BArtifactError("nonfinite model weights")
    return weights, active, iterations


def model_scores(x, weights, active):
    scores = np.asarray(x, dtype=np.float64) @ weights
    scores[:, ~active] = -1e30
    return scores


def selections_for(rankings, losses, sizes, metrics):
    selected = []
    for group, ranking in sorted(rankings.items()):
        for size in sizes:
            for metric in metrics:
                choice = min(ranking[:size], key=lambda c: (losses[c][metric], c))
                selected.append({"group": group, "shortlist_size": size,
                                 "metric": metric, "combination": choice,
                                 "validation_loss": losses[choice][metric]})
    return selected


def audit_selection(rankings, losses, selections, top_q):
    rows = []
    for selection in selections:
        metric = selection["metric"]
        ordered = sorted(losses, key=lambda c: (losses[c][metric], c))
        best, worst = losses[ordered[0]][metric], losses[ordered[-1]][metric]
        selected = losses[selection["combination"]][metric]
        shortlist = rankings[selection["group"]][:selection["shortlist_size"]]
        q = min(top_q, len(ordered))
        omission = min(losses[c][metric] for c in shortlist) - best
        rows.append({**selection, "test_loss": selected, "oracle_loss": best,
                     "worst_loss": worst,
                     "recall_at_top_q": len(set(shortlist) & set(ordered[:q])) / q,
                     "absolute_regret": selected - best,
                     "relative_regret": (selected - best) / abs(best) if best else None,
                     "span_normalized_regret": (selected - best) / (worst - best)
                     if worst > best else None,
                     "absolute_omission": omission,
                     "top_q_boundary_ties": sum(losses[c][metric] == losses[ordered[q - 1]][metric]
                                                for c in ordered)})
    return rows


def run(stage, bundle, base_config, protocol, encoder, seed, root):
    started = time.perf_counter()
    config, spec = load_config(base_config), load_spec(protocol)
    if seed not in spec["projection_seeds"] or encoder not in config["features"]["evaluators"]:
        raise E2BArtifactError("unregistered encoder or projection seed")
    context = {"base_config_sha256": sha256_file(base_config),
               "protocol_sha256": sha256_file(protocol), "encoder": encoder,
               "projection_seed": seed, "source_sha256": code_hashes()}
    upstream = {}
    if stage != "screen":
        upstream["screen"] = check_artifact(root / "screen", context, "screen")
    if stage == "test-audit":
        upstream["validate"] = check_artifact(root / "validate", context, "validate")
        if upstream["validate"]["upstream"] != {"screen": upstream["screen"]["artifact_id"]}:
            raise E2BArtifactError("validation used a different screen")
    # Verify frozen upstream records before opening this stage's feature payload.
    loaded = load_stage_bundle(bundle, base_config, stage, encoder)
    metadata = loaded["metadata"]
    lineage = {k: metadata[k] for k in ("manifest_id", "candidate_set_id",
                                       "source_feature_file_sha256")}
    if any(a["lineage"] != lineage for a in upstream.values()):
        raise E2BArtifactError("stage input lineage mismatch")
    directory = root / stage
    directory.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(config)
    config["features"]["projection"]["seed"] = seed
    projected, block_indices, roles = _projected_views(
        loaded["features"], loaded["sample_ids"], loaded["samples"],
        loaded["candidate_ids"], config, encoder)
    domains = config["dataset"]["domains"]
    classes = int(metadata["class_count"])
    screen = read_json(root / "screen" / "rankings.json") if stage != "screen" else {}
    validation = read_json(root / "validate" / "selections.json") if stage == "test-audit" else {}
    rankings, selected_all, audited_all = {}, {}, []
    expected = int(config["combinations"]["expected_combinations_per_target"])
    progress_path = directory / "progress.json"
    for domain in domains:
        names = sorted(n for n, d in loaded["candidate_domains"].items() if d != domain)
        block_x = {n: projected[block_indices[n]] for n in names}
        target_indices = roles[(domain, TARGET_ROLE[stage])]
        target_x = projected[target_indices]
        if stage == "screen":
            if loaded["labels"] is not None:
                raise E2BArtifactError("screen cannot read labels")
            scores = _score_combinations(block_x, loaded["candidate_domains"], target_x, config)
            groups = {}
            for method, values in scores.items():
                sign = 1 if config["screen"]["methods"][method] == "ascending" else -1
                groups[method] = sorted(values, key=lambda c: (sign * values[c], c))
            combinations = sorted(next(iter(scores.values())))
            for repeat in range(config["screen"]["random_repeats"]):
                rng = np.random.default_rng(stable_seed(config["screen"]["random_seed"],
                                                        encoder, domain, repeat))
                groups[f"random:{repeat}"] = [combinations[i] for i in rng.permutation(len(combinations))]
            if len(combinations) != expected:
                raise E2BArtifactError("combination count mismatch")
            rankings[domain] = groups
            print(f"screen encoder={encoder} seed={seed} domain={domain}", flush=True)
            continue
        labels = loaded["labels"]
        combinations = sorted(next(iter(screen[domain].values())))
        if len(combinations) != expected:
            raise E2BArtifactError("frozen candidate coverage mismatch")
        for readout in spec["readouts"]:
            key = f"{domain}__{readout}"
            losses, fit_records, weights_all, active_all = {}, [], [], []
            if stage == "test-audit":
                with np.load(root / "validate" / f"{key}.npz", allow_pickle=False) as payload:
                    if list(payload["combinations"]) != combinations:
                        raise E2BArtifactError("stored model order mismatch")
                    weights_all, active_all = payload["weights"], payload["active"]
            for i, combination in enumerate(combinations):
                tick = time.perf_counter()
                if stage == "validate":
                    idx = np.concatenate([block_indices[n] for n in parse_combination(combination)])
                    if len(idx) != int(config["combinations"]["fixed_training_sample_count"]):
                        raise E2BArtifactError("unequal source sample cost")
                    weights, active, iterations = fit_readout(projected[idx], labels[idx], classes, readout, spec)
                    weights_all.append(weights)
                    active_all.append(active)
                    fit_records.append({"combination": combination, "iterations": iterations,
                                        "fit_seconds": time.perf_counter() - tick})
                else:
                    weights, active = weights_all[i], active_all[i]
                losses[combination] = prediction_losses(model_scores(target_x, weights, active), labels[target_indices])
                if (i + 1) % 50 == 0 or i + 1 == len(combinations):
                    progress = {"stage": stage, "domain": domain, "readout": readout,
                                "completed_in_group": i + 1, "group_total": len(combinations),
                                "elapsed_seconds": time.perf_counter() - started, "updated_utc": utc_now()}
                    write_json_atomic(progress_path, progress)
                    print(json.dumps(progress), flush=True)
            write_json_atomic(directory / f"{key}_losses.json", losses)
            if stage == "validate":
                np.savez(directory / f"{key}.npz", combinations=np.asarray(combinations),
                         weights=np.asarray(weights_all), active=np.asarray(active_all))
                write_json_atomic(directory / f"{key}_fits.json", {"records": fit_records})
                selected_all[key] = selections_for(screen[domain], losses,
                    config["screen"]["shortlist_sizes"], spec["metrics"])
            else:
                audited_all.extend({"encoder": encoder, "projection_seed": seed,
                                    "target_domain": domain, "readout": readout, **row}
                                   for row in audit_selection(screen[domain], losses, validation[key],
                                                              config["test_audit"]["top_q"]))
    if stage == "screen":
        write_json_atomic(directory / "rankings.json", rankings)
    elif stage == "validate":
        write_json_atomic(directory / "selections.json", selected_all)
    else:
        write_json_atomic(directory / "audit.json", {"rows": audited_all})
    core = {"status": "complete", "schema_version": "e2b-readout-projection-artifact-v1",
            "stage": stage, "context": context, "lineage": lineage,
            "bundle_id": metadata["bundle_id"], "runner_revision": git_revision(),
            "upstream": {k: v["artifact_id"] for k, v in upstream.items()},
            "files": {p.name: sha256_file(p) for p in sorted(directory.iterdir()) if p.is_file()},
            "created_utc": utc_now(), "elapsed_seconds": time.perf_counter() - started,
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "sklearn": sklearn.__version__},
            "interpretation": spec["evaluation"], "cost_claim": spec["cost_claim"],
            "access": {"source_labels": stage != "screen",
                       "target_validation": stage == "validate", "target_test": stage == "test-audit"}}
    artifact = {**core, "artifact_id": canonical_json_sha256(core)}
    write_json_atomic(directory / "manifest.json", artifact)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("screen", "validate", "test-audit"))
    for name in ("stage-bundle", "base-config", "protocol", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = run(args.stage, args.stage_bundle, args.base_config, args.protocol,
                     args.encoder, args.seed, args.output_root)
    print(json.dumps({"status": result["status"], "stage": result["stage"],
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
