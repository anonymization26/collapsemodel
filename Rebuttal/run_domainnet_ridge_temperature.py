"""Single-unit ridge calibration on the archived clipart candidate family."""
import csv
import hashlib
import os
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import softmax
import torch
from torchvision import models

from benchmark_two_stage_encoders_npu import load_dinov2_b14, model_state_sha256
from extract_target_conditioned_e2b_features_npu import ManifestDataset
from metrics.target_conditioned_e2b import jl_project, stable_seed, sha256_file

root, weight = map(Path, sys.argv[1:])
spec = json.loads(Path("pilot.json").read_text())
audit = json.loads(Path("temperature_inputs.json").read_text())
reference = spec["encoder"]
assert sha256_file(weight) == reference["checkpoint_file_sha256"]
encoder = audit["encoder"]
if encoder == "resnet50":
    model = models.resnet50(weights=None)
    model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
    model.fc = torch.nn.Identity()
    transform = models.ResNet50_Weights.IMAGENET1K_V2.transforms()
else:
    assert encoder == "dinov2_b14"
    model, transform, checkpoint = load_dinov2_b14()
    assert Path(checkpoint).resolve() == weight.resolve()
assert model_state_sha256(model) == reference["model_state_sha256"]
assert hashlib.sha256(repr(transform).encode()).hexdigest() == reference["preprocess_sha256"]
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
samples = audit["samples"]
verified = Path("verified_features.npz")
if verified.exists():
    with np.load(verified, allow_pickle=False) as cached:
        features = cached["features"]
else:
    model.eval().cuda()
    loader = torch.utils.data.DataLoader(ManifestDataset(root, samples, transform),
                                        batch_size=64, num_workers=4)
    with torch.inference_mode():
        raw = np.concatenate([model(images.cuda()).cpu().numpy() for images, _ in loader])
    features = jl_project(raw, 128, stable_seed(20260911, encoder)).astype(np.float64)
labels = np.array([int(row["class_id"]) for row in samples])
assert np.all((labels >= 0) & (labels < 100))
indices = {row["sample_id"]: i for i, row in enumerate(samples)}
roles = {role: np.array([i for i, row in enumerate(samples) if row["role"] == role])
         for role in ("target_validation", "target_test")}
validation, test = roles.values()
ordered = sorted(range(len(validation)), key=lambda i: (
    hashlib.sha256(("ridge-temperature-v1:" + samples[validation[i]]["sample_id"]).encode()).digest(),
    samples[validation[i]]["sample_id"]))
calibration, selection = ordered[::2], ordered[1::2]
grams, rhs = {}, {}
for block in spec["blocks"]:
    ix = [indices[key] for key in block["sample_ids"]]
    x = features[ix]
    grams[block["candidate_id"]] = x.T @ x
    rhs[block["candidate_id"]] = x.T @ np.eye(100)[labels[ix]]

def metrics(logits, y, temperature=1):
    probabilities = softmax(logits / temperature, axis=1)
    brier = np.mean(np.sum(probabilities ** 2, axis=1) + 1 -
                    2 * probabilities[np.arange(len(y)), y])
    return float(brier), float(np.mean(logits.argmax(axis=1) == y)), float(probabilities.max(axis=1).mean())

weights, raw_validation = {}, {}
errors = {"validation": [], "test": []}
for candidate in audit["validation"]:
    names = candidate.split("|")
    gram = np.eye(128) + sum(grams[name] for name in names)
    w = np.linalg.solve((gram + gram.T) / 2, sum(rhs[name] for name in names))
    weights[candidate] = w
    raw_validation[candidate] = features[validation] @ w
    for stage, rows in roles.items():
        label = stage.removeprefix("target_")
        brier = metrics(features[rows] @ w, labels[rows])[0]
        errors[label].append(abs(brier - audit[label][candidate]["brier_score"]))
gate = {key: max(values) for key, values in errors.items()}
print(f"T=1 maximum absolute Brier differences: {gate}", flush=True)
assert max(gate.values()) <= 0.0001
Path("t1_gate.json").write_text(json.dumps(gate, indent=2))
if os.getenv("RIDGE_T1_ONLY") == "1":
    np.savez("verified_features.npz", features=features)
    sys.exit(0)
temperatures = np.geomspace(0.01, 100, 81)

def fit_temperature(logits, y):
    return min(temperatures, key=lambda t: (metrics(logits, y, t)[0], abs(np.log(t))))

yv = labels[validation]
calibrated_t = {key: fit_temperature(z[calibration], yv[calibration]) for key, z in raw_validation.items()}
raw_select = {key: metrics(z[selection], yv[selection])[0] for key, z in raw_validation.items()}
cal_select = {key: metrics(z[selection], yv[selection], calibrated_t[key])[0]
              for key, z in raw_validation.items()}
decisions = []
for budget in (10, 25, 50, 100, 227, 455):
    rankings = audit["rankings"] if budget < 455 else {"full": sorted(weights)}
    for method, order in rankings.items():
        prefix = order[:budget]
        original = min(prefix, key=lambda key: (audit["validation"][key]["brier_score"], key))
        raw = min(prefix, key=lambda key: (raw_select[key], key))
        calibrated = min(prefix, key=lambda key: (cal_select[key], key))
        for arm, key, t in (
            ("original", original, 1),
            ("original_calibrated", original, fit_temperature(raw_validation[original], yv)),
            ("split_uncalibrated", raw, 1),
            ("split_calibrated", calibrated, calibrated_t[calibrated]),
        ):
            decisions.append(dict(budget=budget, method=method, arm=arm, candidate=key,
                                  temperature=float(t)))
Path("decisions.json").write_text(json.dumps(decisions, indent=2))
for row in decisions:
    z = features[test] @ weights[row["candidate"]]
    row.update(zip(("brier", "accuracy", "mean_max_probability"),
                   metrics(z, labels[test], row["temperature"])))
with Path("results.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(decisions[0]))
    writer.writeheader()
    writer.writerows(decisions)
print(f"PASS: {len(weights)} ridge fits, {len(validation)} validation, {len(test)} test, "
      f"{len(decisions)} evaluations", flush=True)
