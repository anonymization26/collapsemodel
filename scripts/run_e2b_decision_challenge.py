#!/usr/bin/env python3
"""Role-isolated pool reconstruction and source-domain controls."""

import argparse
import copy
import hashlib
from itertools import combinations
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from run_e2b_readout_projection import (
    ROOT, audit_selection, check_artifact, code_hashes, fit_readout, load_spec,
    model_scores, prediction_losses, selections_for,
)
from run_target_conditioned_e2b_shortlist import _projected_views, utc_now
from metrics.e2b_stage_bundle import load_stage_bundle, TARGET_ROLE
from metrics.target_conditioned_e2b import (
    E2BArtifactError, canonical_json_sha256, load_config, parse_combination,
    read_json, sha256_file, stable_seed, write_json_atomic,
)


def partition_candidates(loaded, construction, seed):
    original = loaded["candidate_ids"]
    domains = loaded["candidate_domains"]
    if construction == "original":
        return original, domains
    if construction != "hash_partition":
        raise E2BArtifactError("unknown candidate construction")
    members, result_domains = {}, {}
    for domain in sorted(set(domains.values())):
        names = sorted(n for n in original if domains[n] == domain)
        sizes = {len(original[n]) for n in names}
        if len(sizes) != 1:
            raise E2BArtifactError("repartition requires equal-size blocks")
        size = sizes.pop()
        ids = [sid for n in names for sid in original[n]]
        if len(set(ids)) != len(ids):
            raise E2BArtifactError("overlapping source blocks")
        ordered = sorted(ids, key=lambda sid: (
            hashlib.sha256(f"{seed}|{domain}|{sid}".encode()).hexdigest(), sid))
        for i in range(len(names)):
            name = f"{domain}__hash_{i}"
            members[name] = ordered[i * size:(i + 1) * size]
            result_domains[name] = domain
    return members, result_domains


def score_method(block_x, domains, target_x, budget, method):
    """Compute only the requested score, including its sufficient statistics."""
    names = sorted(block_x)
    choices = list(combinations(names, budget))
    labels = ["|".join(c) for c in choices]
    if method in ("random", "exhaustive"):
        return dict.fromkeys(labels, 0.0)
    target_x = np.asarray(target_x, dtype=np.float64)
    sizes = {n: len(block_x[n]) for n in names}
    if method == "mean_matching":
        sums = {n: np.sum(block_x[n], axis=0, dtype=np.float64) for n in names}
        target_mean = target_x.mean(axis=0)
        return {label: float(np.sum((sum(sums[n] for n in c) / sum(sizes[n] for n in c)
                                     - target_mean) ** 2)) for label, c in zip(labels, choices)}
    grams = {n: np.asarray(block_x[n], dtype=np.float64).T @ block_x[n] for n in names}
    target_moment = target_x.T @ target_x / len(target_x)
    if method == "domain_moment":
        domain_moments = {d: sum(grams[n] for n in names if domains[n] == d)
                          / sum(sizes[n] for n in names if domains[n] == d)
                          for d in set(domains[n] for n in names)}
    scores = {}
    for label, c in zip(labels, choices):
        count = sum(sizes[n] for n in c)
        if method == "domain_moment":
            moment = sum(sizes[n] * domain_moments[domains[n]] for n in c) / count
            scores[label] = float(np.sum((moment - target_moment) ** 2))
        else:
            gram = sum(grams[n] for n in c)
            if method == "target_a":
                # Cholesky/solve is algebraically identical to the frozen spectral score.
                scores[label] = float(np.trace(np.linalg.solve(np.eye(len(gram)) + gram,
                                                               target_moment)))
            elif method == "second_moment_mmd":
                scores[label] = float(np.sum((gram / count - target_moment) ** 2))
            else:
                raise E2BArtifactError("unknown score")
    return scores


def composition(label, domains):
    return tuple(sorted(domains[n] for n in parse_combination(label)))


def within_composition(rankings, losses, domains, metrics):
    buckets = {}
    for label in losses:
        buckets.setdefault(composition(label, domains), []).append(label)
    rows = []
    for signature, candidates in sorted(buckets.items()):
        if len(candidates) < 2:
            continue
        members = set(candidates)
        size = max(1, len(candidates) // 2)
        for metric in metrics:
            best = min(candidates, key=lambda c: (losses[c][metric], c))
            for group, ranking in rankings.items():
                if group == "exhaustive":
                    continue
                retained = [c for c in ranking if c in members][:size]
                rows.append({"composition": list(signature), "candidate_count": len(candidates),
                             "shortlist_size": size, "metric": metric, "group": group,
                             "oracle_retained": int(best in retained),
                             "absolute_omission": min(losses[c][metric] for c in retained) - losses[best][metric]})
    return rows


def run(stage, bundle, base_config, readout_protocol, protocol, encoder, construction, root):
    started = time.perf_counter()
    config, spec, challenge = load_config(base_config), load_spec(readout_protocol), read_json(protocol)
    if (challenge["schema_version"] != "e2b-decision-cost-v1"
            or construction not in challenge["constructions"]):
        raise E2BArtifactError("unregistered challenge")
    sources = {**code_hashes(), "scripts/run_e2b_decision_challenge.py": sha256_file(Path(__file__))}
    context = {"base_config_sha256": sha256_file(base_config),
               "readout_protocol_sha256": sha256_file(readout_protocol),
               "protocol_sha256": sha256_file(protocol), "encoder": encoder,
               "construction": construction, "source_sha256": sources}
    upstream = {}
    if stage != "screen":
        upstream["screen"] = check_artifact(root / "screen", context, "screen")
    if stage == "test-audit":
        upstream["validate"] = check_artifact(root / "validate", context, "validate")
        if upstream["validate"]["upstream"] != {"screen": upstream["screen"]["artifact_id"]}:
            raise E2BArtifactError("validation lineage mismatch")
    loaded = load_stage_bundle(bundle, base_config, stage, encoder)
    members, domains = partition_candidates(loaded, construction, challenge["partition_seed"])
    membership_id = canonical_json_sha256(members)
    lineage = {k: loaded["metadata"][k] for k in ("manifest_id", "source_feature_file_sha256")}
    lineage["candidate_membership_id"] = membership_id
    if any(a["lineage"] != lineage for a in upstream.values()):
        raise E2BArtifactError("stage construction or source mismatch")
    directory = root / stage
    directory.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(config)
    config["features"]["projection"]["seed"] = challenge["projection_seed"]
    projected, indices, roles = _projected_views(loaded["features"], loaded["sample_ids"],
        loaded["samples"], members, config, encoder)
    classes = int(loaded["metadata"]["class_count"])
    frozen = read_json(root / "screen/rankings.json") if stage != "screen" else {}
    selections = read_json(root / "validate/selections.json") if stage == "test-audit" else {}
    all_rankings, all_selected, audited, conditional = {}, {}, [], []
    for domain in config["dataset"]["domains"]:
        names = sorted(n for n in members if domains[n] != domain)
        block_x = {n: projected[indices[n]] for n in names}
        target_idx = roles[domain, TARGET_ROLE[stage]]
        target_x = projected[target_idx]
        if stage == "screen":
            if loaded["labels"] is not None:
                raise E2BArtifactError("screen labels forbidden")
            groups, score_times = {}, {}
            for method in challenge["methods"]:
                tick = time.perf_counter()
                scores = score_method(block_x, domains, target_x, 3, method)
                score_times[method] = time.perf_counter() - tick
                if method == "random":
                    ordered = sorted(scores)
                    for repeat in range(challenge["random_repeats"]):
                        rng = np.random.default_rng(stable_seed(20260911, encoder, domain, repeat))
                        groups[f"random:{repeat}"] = [ordered[i] for i in rng.permutation(len(ordered))]
                else:
                    groups[method] = sorted(scores, key=lambda c: (scores[c], c))
            groups["exhaustive"] = sorted(scores)
            all_rankings[domain] = groups
            write_json_atomic(directory / f"{domain}_scores_seconds.json", score_times)
            continue
        choices = frozen[domain]["exhaustive"]
        for readout in spec["readouts"]:
            key = f"{domain}__{readout}"
            losses, weights_all, active_all = {}, [], []
            if stage == "test-audit":
                with np.load(root / "validate" / f"{key}.npz", allow_pickle=False) as data:
                    if list(data["combinations"]) != choices:
                        raise E2BArtifactError("model combination order mismatch")
                    weights_all, active_all = data["weights"], data["active"]
            for i, c in enumerate(choices):
                if stage == "validate":
                    idx = np.concatenate([indices[n] for n in parse_combination(c)])
                    if len(idx) != config["combinations"]["fixed_training_sample_count"]:
                        raise E2BArtifactError("source cost mismatch")
                    w, active, _ = fit_readout(projected[idx], loaded["labels"][idx], classes, readout, spec)
                    weights_all.append(w)
                    active_all.append(active)
                else:
                    w, active = weights_all[i], active_all[i]
                losses[c] = prediction_losses(model_scores(target_x, w, active), loaded["labels"][target_idx])
            write_json_atomic(directory / f"{key}_losses.json", losses)
            if stage == "validate":
                np.savez(directory / f"{key}.npz", combinations=np.asarray(choices),
                         weights=np.asarray(weights_all), active=np.asarray(active_all))
                selected = selections_for({g:r for g,r in frozen[domain].items() if g != "exhaustive"},
                    losses, challenge["shortlist_sizes"], challenge["metrics"])
                selected += selections_for({"exhaustive": choices}, losses, [len(choices)], challenge["metrics"])
                all_selected[key] = selected
            else:
                identity = {"encoder": encoder, "projection_seed": challenge["projection_seed"],
                            "target_domain": domain, "readout": readout, "construction": construction}
                audited += [{**identity, **row} for row in audit_selection(frozen[domain], losses,
                    selections[key], config["test_audit"]["top_q"])]
                conditional += [{**identity, **row} for row in within_composition(
                    frozen[domain], losses, domains, challenge["metrics"])]
            print(f"{stage} {encoder} {construction} {domain} {readout} complete", flush=True)
    if stage == "screen":
        write_json_atomic(directory / "rankings.json", all_rankings)
        write_json_atomic(directory / "candidate_membership.json", members)
    elif stage == "validate":
        write_json_atomic(directory / "selections.json", all_selected)
    else:
        write_json_atomic(directory / "audit.json", {"rows": audited})
        write_json_atomic(directory / "within_composition.json", {"rows": conditional})
    core = {"status": "complete", "stage": stage, "context": context, "lineage": lineage,
            "created_utc": utc_now(), "elapsed_seconds": time.perf_counter() - started,
            "upstream": {k: v["artifact_id"] for k,v in upstream.items()},
            "files": {p.name: sha256_file(p) for p in directory.iterdir() if p.is_file()}}
    result = {**core, "artifact_id": canonical_json_sha256(core)}
    write_json_atomic(directory / "manifest.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=list(TARGET_ROLE))
    for name in ("bundle", "base-config", "readout-protocol", "protocol", "output-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--construction", choices=("original", "hash_partition"), required=True)
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a.stage, a.bundle, a.base_config, a.readout_protocol, a.protocol,
            a.encoder, a.construction, a.output_root)
