"""Benchmark exact A-opt over nested half-block inventories from the pilot."""
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from exact_aopt_benchmark import benchmark

spec = json.loads(Path("pilot.json").read_text())
features = np.load("transport_features.npy", allow_pickle=False)
samples = spec["samples"]
assert features.shape == (len(samples), 128)
indices = {row["sample_id"]: i for i, row in enumerate(samples)}
target = features[[i for i, row in enumerate(samples) if row["role"] == "target_selection"]]
moment = target.T @ target / len(target)
half_grams = {}
for block in spec["blocks"]:
    features_in_block = features[[indices[key] for key in block["sample_ids"]]]
    assert len(features_in_block) == 512
    for half, values in enumerate(np.split(features_in_block, 2)):
        half_grams[f'{block["candidate_id"]}:{half}'] = values.T @ values
results = {}
for count in (15, 20, 25, 30):
    blocks = {key: half_grams[key] for key in sorted(half_grams)[:count]}
    combos = {"|".join(combo): combo for combo in combinations(blocks, 3)}
    result = benchmark(blocks, moment, combos)
    for anchor in result["anchors"].values():
        assert all(row["matches_exhaustive"] for row in anchor.values())
    results[count] = result
    print(f"PASS: {count} blocks, {len(combos)} candidates, all exact shortlists match", flush=True)
    Path("scaling.json").write_text(json.dumps(results, indent=2) + "\n")
