"""Reuse each encoder's frozen features across all six transport comparisons."""
import hashlib
from itertools import combinations
import json
from pathlib import Path
import runpy
import os
import sys
from time import perf_counter

import numpy as np
import torch
from torchvision import models

from benchmark_two_stage_encoders_npu import load_dinov2_b14, model_state_sha256
from exact_aopt_benchmark import _score, _target_factor
from extract_target_conditioned_e2b_features_npu import ManifestDataset
from metrics.target_conditioned_e2b import jl_project, stable_seed

job = Path.cwd()
dataset, weights_dir = map(Path, sys.argv[1:3])
runner = Path(sys.argv[3]).resolve()
spec = json.loads((job / "full_screen_spec.json").read_text())
samples = spec["samples"]
assert len(samples) == len({row["sample_id"] for row in samples}) == 18432
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
domains = sorted({row["domain"] for row in samples})
assert len(domains) == 6

def encode(encoder, reference):
    if encoder == "resnet50":
        weights = weights_dir / "resnet50-11ad3fa6.pth"
        model = models.resnet50(weights=None)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        model.fc = torch.nn.Identity()
        transform = models.ResNet50_Weights.IMAGENET1K_V2.transforms()
    else:
        assert encoder == "dinov2_b14"
        model, transform, checkpoint = load_dinov2_b14()
        weights = Path(checkpoint)
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == reference["checkpoint_file_sha256"]
    assert model_state_sha256(model) == reference["model_state_sha256"]
    assert hashlib.sha256(repr(transform).encode()).hexdigest() == reference["preprocess_sha256"]
    start = perf_counter()
    model.eval().cuda()
    loader = torch.utils.data.DataLoader(ManifestDataset(dataset, samples, transform),
                                        batch_size=64, num_workers=4)
    with torch.inference_mode():
        raw = np.concatenate([model(images.cuda()).cpu().numpy() for images, _ in loader])
    features = jl_project(raw, 128, stable_seed(20260911, encoder)).astype(np.float64)
    print(f"{encoder}: verified and encoded {len(raw)} frozen rows in {perf_counter()-start:.3f} s", flush=True)
    return features

def baseline(blocks, target, method):
    start = perf_counter()
    grams = {name: x.T @ x for name, x in blocks.items()}
    moment = target.T @ target / len(target)
    factor = _target_factor(moment) if method == "target_a" else None
    prior = np.eye(moment.shape[0])
    scores = {}
    for combo in combinations(sorted(grams), 3):
        parts = [grams[block] for block in combo]
        if method == "target_a":
            value = _score(parts, prior, factor)
        else:
            value = float(np.sum((sum(parts) / 1536 - moment) ** 2))
        scores["|".join(combo)] = value
    ranked = sorted(scores, key=lambda key: (scores[key], key))
    return ranked, perf_counter() - start, scores

for encoder, reference in spec["encoders"].items():
    features = encode(encoder, reference)
    for domain in domains:
        blocks = [block for block in spec["blocks"] if block["domain"] != domain]
        ids = {key for block in blocks for key in block["sample_ids"]}
        rows = [i for i, row in enumerate(samples) if row["sample_id"] in ids
                or (row["role"] == "target_selection" and row["domain"] == domain)]
        unit = job / "units" / encoder / domain
        unit.mkdir(parents=True, exist_ok=False)
        assert len(rows) == 9216
        (unit / "pilot.json").write_text(json.dumps(dict(encoder=reference, blocks=blocks,
                                                         samples=[samples[i] for i in rows])))
        np.save(unit / "transport_features.npy", features[rows])
        os.chdir(unit)
        saved = sys.argv[:]
        try:
            sys.argv = [str(runner)]
            local = runpy.run_path(str(runner), run_name="__main__")
        finally:
            sys.argv = saved
            os.chdir(job)
        path = unit / "screen.json"
        screen = json.loads(path.read_text())
        screen["geometry_baselines"] = {}
        for method in ("target_a", "second_moment_mmd"):
            trials = [baseline(local["blocks"], local["target"], method) for _ in range(3)]
            ranked, _, scores = trials[0]
            assert all(order == ranked for order, _, _ in trials)
            times = [elapsed for _, elapsed, _ in trials]
            screen["timing"][method] = dict(seconds=times, median_seconds=float(np.median(times)))
            screen["geometry_baselines"][method] = dict(ranking=ranked, scores=scores)
        path.write_text(json.dumps(screen, indent=2) + "\n")
        print(f"PASS: {encoder}/{domain} has 455 frozen transport candidates", flush=True)
