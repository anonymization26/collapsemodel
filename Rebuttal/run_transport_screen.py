"""Extract or reuse frozen pilot features and score fixed candidate unions."""
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
from time import perf_counter
import warnings

import numpy as np

work = Path.cwd()
spec = json.loads((work / "pilot.json").read_text())
encoder = spec["encoder"]["name"]
samples = spec["samples"]
cache = work / "transport_features.npy"
if cache.exists():
    features = np.load(cache, allow_pickle=False)
else:
    import torch
    from torchvision import models
    from benchmark_two_stage_encoders_npu import load_dinov2_b14, model_state_sha256
    from extract_target_conditioned_e2b_features_npu import ManifestDataset
    from metrics.target_conditioned_e2b import jl_project, stable_seed

    root, weight = map(Path, sys.argv[1:])
    reference = spec["encoder"]
    assert hashlib.sha256(weight.read_bytes()).hexdigest() == reference["checkpoint_file_sha256"]
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
    model.eval().cuda()
    loader = torch.utils.data.DataLoader(ManifestDataset(root, samples, transform),
                                        batch_size=64, num_workers=4)
    start = perf_counter()
    with torch.inference_mode():
        raw = np.concatenate([model(images.cuda()).cpu().numpy() for images, _ in loader])
    features = jl_project(raw, 128, stable_seed(20260911, encoder)).astype(np.float64)
    np.save(cache, features)
    print(f"Verified and encoded {len(features)} rows in {perf_counter() - start:.3f} s", flush=True)
assert features.shape == (len(samples), 128)
indices = {row["sample_id"]: i for i, row in enumerate(samples)}
assert len(indices) == len(samples)
blocks = {row["candidate_id"]: features[[indices[key] for key in row["sample_ids"]]]
          for row in spec["blocks"]}
target_rows = [i for i, row in enumerate(samples) if row["role"] == "target_selection"]
target = features[target_rows]
domain, = {samples[i]["domain"] for i in target_rows}
assert len(target) == 1536 and len(blocks) == 15
assert all(len(block) == 512 for block in blocks.values())
combos = {"|".join(combo): combo for combo in combinations(sorted(blocks), 3)}
directions = np.random.default_rng(45649).normal(size=(128, 32))
directions /= np.linalg.norm(directions, axis=0)

def score_transport(whiten):
    start = perf_counter()
    transform = directions
    if whiten:
        source = np.concatenate(list(blocks.values()))
        centered = source - source.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(source))
        ridge = 0.1 * eigenvalues.mean()
        transform = eigenvectors @ ((eigenvectors.T @ directions) / np.sqrt(eigenvalues + ridge)[:, None])
    projected = {name: block @ transform for name, block in blocks.items()}
    target_sorted = np.sort(target @ transform, axis=0)
    scores = {}
    for name, combo in combos.items():
        source_sorted = np.sort(np.concatenate([projected[block] for block in combo]), axis=0)
        scores[name] = float(np.mean((source_sorted - target_sorted) ** 2))
    ranked = sorted(scores, key=lambda key: (scores[key], key))
    return ranked, perf_counter() - start

screen = dict(encoder=encoder, domain=domain, transport={}, timing={}, exact={})
for method, whiten in (("sliced_w2", False), ("whitened_sliced_w2", True)):
    trials = [score_transport(whiten) for _ in range(3)]
    ranked, _ = trials[0]
    assert all(order == ranked for order, _ in trials)
    screen["transport"][method] = ranked
    seconds = [duration for _, duration in trials]
    screen["timing"][method] = dict(seconds=seconds, median_seconds=float(np.median(seconds)))
    print(f"{method}: {np.median(seconds):.4f} s median", flush=True)
grams = {name: block.T @ block for name, block in blocks.items()}
moment = target.T @ target / len(target)
if len(sys.argv) > 3 and sys.argv[3] == "--exact":
    from exact_aopt_benchmark import benchmark
    screen["exact"] = benchmark(grams, moment, combos)
else:
    warnings.warn("Exact benchmark deferred")
(work / "screen.json").write_text(json.dumps(screen, indent=2) + "\n")
