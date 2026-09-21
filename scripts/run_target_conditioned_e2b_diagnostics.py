#!/usr/bin/env python3
"""Post-selection, conditional test-sample diagnostics; never reselect models."""

import argparse
import itertools
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from metrics.target_conditioned_e2b import (
    E2BArtifactError, all_combinations, canonical_json_sha256, combination_name,
    hash_ids, load_config, parse_combination, read_json, sha256_file, stable_seed,
    write_csv_atomic, write_json_atomic,
)
from run_target_conditioned_e2b_shortlist import (
    METRIC_FIELDS, _read_csv, _screen_groups, _spearman, _stage_inputs,
    _validate_screen, _validate_validation, git_revision, utc_now,
)
from summarize_target_conditioned_e2b import _validate_audit

METRICS = ("brier_score", "squared_loss", "nll", "error_rate")


def sample_losses(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if (scores.ndim != 2 or len(truth) != len(scores) or not len(truth)
            or not np.isfinite(scores).all() or np.any(truth < 0)
            or np.any(truth >= scores.shape[1])):
        raise E2BArtifactError("invalid prediction arrays")
    one_hot = np.eye(scores.shape[1], dtype=np.float64)[truth]
    probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return {
        "brier_score": np.sum((probabilities - one_hot) ** 2, axis=1),
        "squared_loss": np.sum((scores - one_hot) ** 2, axis=1),
        "nll": -np.log(np.maximum(probabilities[np.arange(len(truth)), truth], 1e-15)),
        "error_rate": (probabilities.argmax(axis=1) != truth).astype(np.float64),
    }


def fit_sample_losses(combinations, block_features, block_labels, target, truth,
                      class_count, regularization):
    dimension = next(iter(block_features.values())).shape[1]
    identity = np.eye(dimension, dtype=np.float64)
    one_hot = np.eye(class_count, dtype=np.float64)
    grams, rhs_blocks = {}, {}
    for name, value in block_features.items():
        matrix = np.asarray(value, dtype=np.float64)
        grams[name] = matrix.T @ matrix
        rhs_blocks[name] = matrix.T @ one_hot[block_labels[name]]
    target = np.asarray(target, dtype=np.float64)
    losses = {metric: np.empty((len(combinations), len(truth)), dtype=np.float64)
              for metric in METRICS}
    for index, label in enumerate(combinations):
        names = parse_combination(label)
        gram = regularization * identity + sum(
            (grams[name] for name in names), start=np.zeros_like(identity))
        rhs = sum((rhs_blocks[name] for name in names),
                  start=np.zeros((dimension, class_count), dtype=np.float64))
        weights = np.linalg.solve((gram + gram.T) / 2.0, rhs)
        for metric, values in sample_losses(target @ weights, truth).items():
            losses[metric][index] = values
    return losses


def bootstrap_topq(losses, shortlist_masks, top_q, repeats, batch_size, seed):
    """All combinations share draws; callers reuse the seed and sample order."""
    losses = np.asarray(losses, dtype=np.float64)
    masks = np.asarray(shortlist_masks, dtype=bool)
    if (losses.ndim != 2 or not losses.shape[1] or not np.isfinite(losses).all()
            or masks.ndim != 2 or masks.shape[1] != losses.shape[0]
            or not 1 <= top_q <= losses.shape[0] or repeats < 1 or batch_size < 1):
        raise E2BArtifactError("invalid bootstrap inputs")
    means = losses.mean(axis=1)
    centered = losses - means[:, None]
    empirical = np.argsort(means, kind="stable")[:top_q]
    observed_mask = np.zeros(len(losses), dtype=bool)
    observed_mask[empirical] = True
    indices = np.empty((repeats, top_q), dtype=np.int64)
    recalls = np.empty((repeats, len(masks)), dtype=np.float64)
    overlaps = np.empty(repeats, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample_count = losses.shape[1]
    probabilities = np.full(sample_count, 1.0 / sample_count)
    for start in range(0, repeats, batch_size):
        stop = min(start + batch_size, repeats)
        counts = rng.multinomial(sample_count, probabilities, size=stop - start)
        resampled = counts @ centered.T / sample_count + means
        top = np.argsort(resampled, axis=1, kind="stable")[:, :top_q]
        indices[start:stop] = top
        overlaps[start:stop] = observed_mask[top].mean(axis=1)
        recalls[start:stop] = masks[:, top].mean(axis=2).T
    return {
        "indices": indices, "recalls": recalls, "overlaps": overlaps,
        "frequency": np.bincount(indices.ravel(), minlength=len(losses)) / repeats,
        "empirical": empirical,
    }


def verify_upstream(config_path, stage_bundle, encoder, screen_dir, validation_dir, audit_dir):
    # Only metadata is opened until the complete upstream chain is verified.
    header = read_json(stage_bundle / "manifest.json")
    feature_hash = str(header["source_feature_file_sha256"])
    screen, screen_rows = _validate_screen(
        screen_dir, config_path, str(header["candidate_set_id"]), feature_hash, encoder)
    validation, _, _ = _validate_validation(
        validation_dir, config_path, screen, feature_hash, encoder)
    audit_encoder, _ = _validate_audit(audit_dir, config_path)
    audit = read_json(audit_dir / "manifest.json")
    expected = {
        "manifest_id": header["manifest_id"],
        "candidate_set_id": header["candidate_set_id"],
        "feature_file_sha256": feature_hash,
        "screen_id": screen["screen_id"],
        "validation_id": validation["validation_id"],
        "validation_manifest_file_sha256": sha256_file(validation_dir / "manifest.json"),
        "validation_selections_file_sha256": sha256_file(validation_dir / "selections.csv"),
        "combinations_file_sha256": sha256_file(audit_dir / "combinations.csv"),
    }
    if audit_encoder != encoder or any(audit.get(k) != v for k, v in expected.items()):
        raise E2BArtifactError("test audit upstream identity/hash mismatch")
    if any(item.get("input_isolation", {}).get("mode") != "role-separated-files"
           for item in (screen, validation, audit)):
        raise E2BArtifactError("diagnostics require isolated upstream stages")
    if audit["input_isolation"].get("bundle_id") != header.get("bundle_id"):
        raise E2BArtifactError("test audit bundle mismatch")
    fields = METRIC_FIELDS + ("test_rank", "in_test_top_q", "normalized_regret")
    rows = _read_csv(audit_dir / "combinations.csv", fields)
    if len(rows) != audit.get("combination_row_count"):
        raise E2BArtifactError("test audit combination coverage mismatch")
    return audit, screen_rows, rows


def distribution(values, confidence):
    tail = (1.0 - confidence) / 2.0
    return {"mean": float(np.mean(values)),
            "p_lower": float(np.quantile(values, tail)),
            "p_upper": float(np.quantile(values, 1.0 - tail))}


def run_diagnostics(config_path, diagnostics_path, stage_bundle, encoder,
                    screen_dir, validation_dir, audit_dir, output_dir):
    started, started_utc = time.perf_counter(), utc_now()
    if output_dir.exists():
        raise E2BArtifactError("refusing to overwrite diagnostics")
    config, settings = load_config(config_path), read_json(diagnostics_path)
    if (settings.get("schema_version") != "e2b-post-selection-diagnostics-v1"
            or settings.get("metrics") != list(METRICS)
            or settings.get("refit_within_bootstrap") is not False
            or settings.get("change_selection_or_hyperparameters") is not False
            or settings.get("resampling_unit") != "target-test-sample"
            or settings.get("paired_across_combinations_and_encoders_within_target_domain") is not True
            or settings.get("aggregate_sample_bootstrap_as_cross_domain_evidence") is not False
            or not 0 < float(settings["confidence_level"]) < 1):
        raise E2BArtifactError("unsupported diagnostics protocol")
    top_q = int(settings["top_q"])
    if top_q != int(config["test_audit"]["top_q"]):
        raise E2BArtifactError("top-q must match the original audit")
    audit, screen_rows, audit_rows = verify_upstream(
        config_path, stage_bundle, encoder, screen_dir, validation_dir, audit_dir)
    (manifest, _, _, sample_ids, labels, candidate_domains, projected,
     block_indices, role_indices, isolation) = _stage_inputs(
        "test-audit", None, config_path, None, None, None, None, None, encoder, stage_bundle)
    if labels is None:
        raise E2BArtifactError("test audit labels missing")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output_dir / "status.json", {"status": "running", "started_utc": started_utc})
    reference = {(row["target_domain"], row["combination"]): row for row in audit_rows}
    if len(reference) != len(audit_rows):
        raise E2BArtifactError("duplicate audit combinations")
    frequency_rows, stability_rows, recall_rows, metric_rows, domains = [], [], [], [], []
    visited = set()
    for domain in config["dataset"]["domains"]:
        domain_started = time.perf_counter()
        names = sorted(name for name, source in candidate_domains.items() if source != domain)
        combinations = sorted(combination_name(c) for c in all_combinations(
            names, int(config["combinations"]["budget_blocks"])))
        if len(combinations) != int(config["combinations"]["expected_combinations_per_target"]):
            raise E2BArtifactError("unexpected combination count")
        rows = role_indices[(domain, "target_test")]
        rows = rows[np.argsort(sample_ids[rows], kind="stable")]
        test_ids = [str(value) for value in sample_ids[rows]]
        sample_hash = hash_ids(test_ids)
        seed = stable_seed(settings["bootstrap_seed"], domain, sample_hash)
        losses = fit_sample_losses(
            combinations, {name: projected[block_indices[name]] for name in names},
            {name: labels[block_indices[name]] for name in names}, projected[rows], labels[rows],
            int(manifest["class_count"]), float(config["validation"]["ridge_regularization"]))
        max_error = 0.0
        for index, name in enumerate(combinations):
            key = (domain, name)
            expected = reference[key]
            if expected["encoder"] != encoder:
                raise E2BArtifactError("audit row encoder mismatch")
            visited.add(key)
            for metric in METRICS:
                value = 1.0 - float(expected["accuracy"]) if metric == "error_rate" else float(expected[metric])
                error = abs(float(losses[metric][index].mean()) - value)
                max_error = max(max_error, error)
                if not np.isfinite(error) or error > 1e-10:
                    raise E2BArtifactError(f"per-sample loss mismatch: {domain} {name} {metric}: {error}")
        print(f"encoder={encoder} target={domain} loss_check_max_error={max_error:.3g}", flush=True)
        groups = _screen_groups(screen_rows, domain)
        lookup = {name: index for index, name in enumerate(combinations)}
        shortlist_keys, masks = [], []
        for (method, repeat), ranked in sorted(groups.items()):
            if len(ranked) != len(combinations) or set(ranked) != set(combinations):
                raise E2BArtifactError("screen combination coverage mismatch")
            for size in config["screen"]["shortlist_sizes"]:
                mask = np.zeros(len(combinations), dtype=bool)
                mask[[lookup[name] for name in ranked[:int(size)]]] = True
                masks.append(mask)
                shortlist_keys.append((method, repeat, int(size)))
        means = {metric: values.mean(axis=1) for metric, values in losses.items()}
        top_sets = {metric: set(np.argsort(values, kind="stable")[:top_q]) for metric, values in means.items()}
        for left, right in itertools.combinations(METRICS, 2):
            metric_rows.append({"encoder": encoder, "target_domain": domain,
                                "left_metric": left, "right_metric": right,
                                "spearman": _spearman(means[left], means[right]),
                                "top_q_overlap": len(top_sets[left] & top_sets[right]) / top_q})
        loss_path, bootstrap_path = output_dir / f"{domain}_losses.npz", output_dir / f"{domain}_bootstrap.npz"
        np.savez_compressed(loss_path, combinations=np.asarray(combinations),
                            sample_ids=np.asarray(test_ids), **losses)
        bootstrap_arrays = {}
        for metric in METRICS:
            boot = bootstrap_topq(losses[metric], masks, top_q, int(settings["bootstrap_repeats"]),
                                  int(settings["bootstrap_batch_size"]), seed)
            empirical = boot["empirical"]
            ranks = np.empty(len(combinations), dtype=int)
            ranks[np.argsort(means[metric], kind="stable")] = np.arange(1, len(combinations) + 1)
            for index, name in enumerate(combinations):
                frequency_rows.append({"encoder": encoder, "target_domain": domain, "metric": metric,
                                       "combination": name, "empirical_loss": means[metric][index],
                                       "empirical_rank": ranks[index], "bootstrap_top_q_frequency": boot["frequency"][index]})
            boundary = np.sort(means[metric], kind="stable")[top_q - 1]
            stability_rows.append({"encoder": encoder, "target_domain": domain, "metric": metric,
                                   "test_sample_count": len(test_ids), "top_q": top_q,
                                   "exact_ties_at_boundary": int(np.count_nonzero(means[metric] == boundary)),
                                   **distribution(boot["overlaps"], float(settings["confidence_level"]))})
            for index, (method, repeat, size) in enumerate(shortlist_keys):
                recall_rows.append({"encoder": encoder, "target_domain": domain, "metric": metric,
                                    "method": method, "repeat": repeat, "shortlist_size": size,
                                    "empirical_recall": float(masks[index][empirical].mean()),
                                    **distribution(boot["recalls"][:, index], float(settings["confidence_level"]))})
            bootstrap_arrays[metric + "_top_indices"] = boot["indices"]
            bootstrap_arrays[metric + "_shortlist_recalls"] = boot["recalls"]
            print(f"encoder={encoder} target={domain} metric={metric} bootstrap={settings['bootstrap_repeats']}", flush=True)
        np.savez_compressed(bootstrap_path, combinations=np.asarray(combinations),
                            shortlist_keys=np.asarray(shortlist_keys, dtype=str), **bootstrap_arrays)
        domains.append({"target_domain": domain, "sample_ids_sha256": sample_hash,
                        "sample_count": len(test_ids), "bootstrap_seed": seed,
                        "loss_mean_max_abs_error": max_error,
                        "loss_file": loss_path.name, "loss_sha256": sha256_file(loss_path),
                        "bootstrap_file": bootstrap_path.name, "bootstrap_sha256": sha256_file(bootstrap_path),
                        "elapsed_seconds": time.perf_counter() - domain_started})
        write_json_atomic(output_dir / "status.json", {"status": "running", "completed_domains": domains})
    if visited != set(reference):
        raise E2BArtifactError("audit contains unexpected combinations or target domains")
    files = {}
    for name, values in (("topq_frequency.csv", frequency_rows), ("topq_stability.csv", stability_rows),
                         ("shortlist_recall.csv", recall_rows), ("metric_agreement.csv", metric_rows)):
        write_csv_atomic(output_dir / name, tuple(values[0]), values)
        files[name] = sha256_file(output_dir / name)
    core = {
        "schema_version": "e2b-diagnostics-result-v1", "encoder": encoder,
        "config_file_sha256": sha256_file(config_path), "diagnostics_settings": settings,
        "diagnostics_file_sha256": sha256_file(diagnostics_path), "test_audit_id": audit["test_audit_id"],
        "test_audit_manifest_sha256": sha256_file(audit_dir / "manifest.json"),
        "input_isolation": isolation, "test_outcomes_change_selection": False,
        "tie_break": "ascending-loss-then-lexicographic-combination",
        "quantile_interpretation": "conditional bootstrap distribution percentiles, not confidence intervals for generalization",
        "domains": domains, "files": files, "base_git_revision": git_revision(),
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in (
            Path(__file__), ROOT / "scripts/run_target_conditioned_e2b_shortlist.py",
            ROOT / "scripts/summarize_target_conditioned_e2b.py",
            ROOT / "code/metrics/target_conditioned_e2b.py", ROOT / "code/metrics/e2b_stage_bundle.py")},
        "runtime": {"started_utc": started_utc, "finished_utc": utc_now(),
                    "elapsed_seconds": time.perf_counter() - started,
                    "numpy_version": np.__version__, "python_version": platform.python_version(), "device": "cpu"},
    }
    result = {**core, "diagnostics_id": canonical_json_sha256(core)}
    write_json_atomic(output_dir / "manifest.json", result)
    write_json_atomic(output_dir / "status.json", {"status": "complete", "diagnostics_id": result["diagnostics_id"]})
    return result


def verify_pairing(directories, output_path):
    if len(directories) != 2:
        raise E2BArtifactError("pairing verification requires two encoders")
    manifests = []
    for directory in directories:
        manifest = read_json(directory / "manifest.json")
        core = {k: v for k, v in manifest.items() if k != "diagnostics_id"}
        if manifest.get("diagnostics_id") != canonical_json_sha256(core):
            raise E2BArtifactError("diagnostics identity mismatch")
        for filename, checksum in manifest["files"].items():
            if sha256_file(directory / filename) != checksum:
                raise E2BArtifactError("diagnostics CSV hash mismatch")
        for domain in manifest["domains"]:
            for prefix in ("loss", "bootstrap"):
                if sha256_file(directory / domain[prefix + "_file"]) != domain[prefix + "_sha256"]:
                    raise E2BArtifactError("diagnostics array hash mismatch")
        manifests.append(manifest)
    left, right = manifests
    keys = ("target_domain", "sample_ids_sha256", "sample_count", "bootstrap_seed")
    if (left["encoder"] == right["encoder"] or left["config_file_sha256"] != right["config_file_sha256"]
            or left["diagnostics_file_sha256"] != right["diagnostics_file_sha256"]
            or left["runtime"]["numpy_version"] != right["runtime"]["numpy_version"]
            or [{k: d[k] for k in keys} for d in left["domains"]] != [{k: d[k] for k in keys} for d in right["domains"]]):
        raise E2BArtifactError("encoders did not use identical paired resampling inputs")
    write_json_atomic(output_path, {"status": "paired-inputs-verified", "created_utc": utc_now(),
                                   "diagnostics_ids": [m["diagnostics_id"] for m in manifests]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    for name in ("config", "diagnostics-config", "stage-bundle", "screen-dir", "validation-dir", "audit-dir", "output-dir"):
        run.add_argument("--" + name, type=Path, required=True)
    run.add_argument("--encoder", required=True)
    verify = commands.add_parser("verify-pairing")
    verify.add_argument("--diagnostics-dir", type=Path, action="append", required=True)
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify-pairing":
        verify_pairing(args.diagnostics_dir, args.output)
    else:
        run_diagnostics(args.config, args.diagnostics_config, args.stage_bundle, args.encoder,
                        args.screen_dir, args.validation_dir, args.audit_dir, args.output_dir)


if __name__ == "__main__":
    main()
