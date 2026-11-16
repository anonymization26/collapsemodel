"""Rebuild the clipart/ResNet-50 screen from frozen, hash-checked images."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torchvision import models

from benchmark_two_stage_encoders_npu import model_state_sha256
from extract_target_conditioned_e2b_features_npu import ManifestDataset
from metrics.target_conditioned_e2b import jl_project, stable_seed
from run_target_conditioned_e2b_shortlist import _score_combinations

root, weight = map(Path, sys.argv[1:])
spec = json.loads(Path("pilot.json").read_text())
reference = spec["encoder"]
assert hashlib.sha256(weight.read_bytes()).hexdigest() == reference["checkpoint_file_sha256"]
model = models.resnet50(weights=None)
model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
model.fc = torch.nn.Identity()
transform = models.ResNet50_Weights.IMAGENET1K_V2.transforms()
assert model_state_sha256(model) == reference["model_state_sha256"]
assert hashlib.sha256(repr(transform).encode()).hexdigest() == reference["preprocess_sha256"]
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
model.eval().cuda()
dataset = ManifestDataset(root, spec["samples"], transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4)
with torch.inference_mode():
    raw = np.concatenate([model(images.cuda()).cpu().numpy() for images, _ in loader])
print(f"Verified and encoded {len(raw)} frozen images", flush=True)
config = spec["config"]
projection = config["features"]["projection"]
features = jl_project(raw, projection["output_dimension"], stable_seed(projection["seed"], "resnet50"))
indices = {row["sample_id"]: i for i, row in enumerate(spec["samples"])}
blocks = {row["candidate_id"]: features[[indices[key] for key in row["sample_ids"]]] for row in spec["blocks"]}
domains = {row["candidate_id"]: row["domain"] for row in spec["blocks"]}
target = features[[i for i, row in enumerate(spec["samples"]) if row["role"] == "target_selection"]]
scores = _score_combinations(blocks, domains, target, config)
print("SCORES_JSON=" + json.dumps({key: scores[key] for key in ("target_a", "target_energy", "second_moment_mmd")}, sort_keys=True))
