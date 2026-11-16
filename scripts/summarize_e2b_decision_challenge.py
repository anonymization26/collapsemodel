#!/usr/bin/env python3
"""Verify completed candidate-construction audits and collapse dependent repeats."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from analyze_e2b_decision_budget import aggregate_rows, digest, dump_csv, mean


def summarize(root, output):
    spec = json.loads((root / "protocol.json").read_text())
    all_curves, all_failures, conditional, inputs = [], [], [], {}
    for construction in spec["constructions"]:
        raw, within = [], []
        for encoder in ("resnet50", "dinov2_b14"):
            job = root / encoder / construction
            manifests = {}
            for stage in ("screen", "validate", "test-audit"):
                directory = job / stage
                path = directory / "manifest.json"
                manifest = json.loads(path.read_text())
                if (manifest["status"] != "complete" or manifest["stage"] != stage
                        or manifest["context"]["encoder"] != encoder
                        or manifest["context"]["construction"] != construction
                        or manifest["context"]["protocol_sha256"] != digest(root / "protocol.json")):
                    raise ValueError("challenge identity mismatch")
                core = {k:v for k,v in manifest.items() if k != "artifact_id"}
                import hashlib
                sealed = hashlib.sha256(json.dumps(core, sort_keys=True, ensure_ascii=True,
                                                   separators=(",", ":")).encode()).hexdigest()
                if sealed != manifest["artifact_id"]:
                    raise ValueError("challenge manifest seal mismatch")
                for name, expected in manifest["files"].items():
                    if name.endswith(".npz"):
                        continue  # Models were checked by the remote audit before test access.
                    if digest(directory / name) != expected:
                        raise ValueError("challenge numerical hash mismatch")
                inputs[str(path)] = digest(path)
                manifests[stage] = manifest
            if manifests["validate"]["upstream"] != {"screen": manifests["screen"]["artifact_id"]}:
                raise ValueError("validation lineage mismatch")
            if manifests["test-audit"]["upstream"] != {s: manifests[s]["artifact_id"] for s in ("screen", "validate")}:
                raise ValueError("test lineage mismatch")
            rows = json.loads((job / "test-audit/audit.json").read_text())["rows"]
            references = {(r["target_domain"], r["readout"], r["metric"]): r["test_loss"]
                          for r in rows if r["group"] == "exhaustive"}
            if len(references) != 36:
                raise ValueError("missing exhaustive validation references")
            for r in rows:
                full = references[r["target_domain"], r["readout"], r["metric"]]
                raw.append({**r, "method": r["group"].split(":")[0],
                            "full_validation_test_loss": full, "gap_to_full": r["test_loss"] - full})
            within += json.loads((job / "test-audit/within_composition.json").read_text())["rows"]
        _, curves, failures = aggregate_rows(raw, spec["absolute_gap_thresholds"])
        all_curves += [{"construction": construction, **r} for r in curves]
        all_failures += [{"construction": construction, **r} for r in failures]
        # Random repeat -> composition -> encoder -> target domain, with equal domain weights.
        levels = [("encoder", "target_domain", "readout", "metric", "composition", "method"),
                  ("encoder", "target_domain", "readout", "metric", "method"),
                  ("target_domain", "readout", "metric", "method"),
                  ("readout", "metric", "method")]
        values = [{**r, "composition": tuple(r["composition"]), "method": r["group"].split(":")[0]}
                  for r in within]
        for fields in levels:
            buckets = defaultdict(list)
            for r in values:
                buckets[tuple(r[k] for k in fields)].append(r)
            values = [{**dict(zip(fields,key)), **{field: mean([r[field] for r in rows])
                       for field in ("oracle_retained", "absolute_omission")}}
                      for key, rows in sorted(buckets.items())]
        conditional += [{"construction": construction, **r} for r in values]
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "complete", "curves": all_curves, "failure_rates": all_failures,
              "within_composition": conditional, "input_sha256": inputs,
              "source_sha256": digest(Path(__file__)), "independent_dataset": False,
              "interpretation": "same-images-independent-label-free-construction-and-domain-composition-controls",
              "models": "remote-frozen-model-hashes-checked-before-test-not-copied-locally"}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for name in ("curves", "failure_rates", "within_composition"):
        dump_csv(output / f"{name}.csv", result[name])
    print(json.dumps({"status": "complete", "curves": len(all_curves), "conditional": len(conditional)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    a = p.parse_args()
    summarize(a.input_root, a.output_root)
