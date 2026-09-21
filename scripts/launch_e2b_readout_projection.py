#!/usr/bin/env python3
"""Launch a bounded, resumeless supplementary sweep in a new output directory."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from run_e2b_readout_projection import ROOT, code_hashes, load_spec
from metrics.target_conditioned_e2b import load_config, sha256_file, write_json_atomic
from run_target_conditioned_e2b_shortlist import git_revision, utc_now


def launch(work, output, workers):
    if not 1 <= workers <= 4:
        raise ValueError("concurrency must be between one and four")
    base = ROOT / "code/configs/target_conditioned_e2b/domainnet_v1.json"
    protocol = ROOT / "code/configs/target_conditioned_e2b/readout_projection_v1.json"
    spec, config = load_spec(protocol), load_config(base)
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    shutil.copyfile(base, output / "base_config.json")
    shutil.copyfile(protocol, output / "protocol.json")
    jobs = [(encoder, seed) for seed in spec["projection_seeds"]
            for encoder in config["features"]["evaluators"]]
    record = {"started_utc": utc_now(), "status": "running", "workers": workers,
              "pid": os.getpid(), "runner_revision": git_revision(),
              "source_sha256": {**code_hashes(),
                  "scripts/launch_e2b_readout_projection.py": sha256_file(Path(__file__))},
              "base_config_sha256": sha256_file(base), "protocol_sha256": sha256_file(protocol),
              "job_count": len(jobs), "readouts": spec["readouts"], "jobs": {},
              "cpu_threads_per_worker": 1,
              "cgroup": {p.name: p.read_text().strip() for p in [
                  Path("/sys/fs/cgroup/cpu.max"), Path("/sys/fs/cgroup/memory.max"),
                  Path("/sys/fs/cgroup/memory.current")] if p.exists()},
              "disk_free_bytes": shutil.disk_usage(work).free}
    write_json_atomic(output / "status.json", record)
    stopped = threading.Event()
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}

    def execute(encoder, seed):
        name = f"{encoder}__{seed}"
        if stopped.is_set():
            return name, {"status": "not_started_after_failure"}
        started = time.perf_counter()
        for stage in ("screen", "validate", "test-audit"):
            command = [sys.executable, str(ROOT / "scripts/run_e2b_readout_projection.py"), stage,
                       "--stage-bundle", str(work / "stage_bundles" / encoder / stage),
                       "--base-config", str(base), "--protocol", str(protocol),
                       "--encoder", encoder, "--seed", str(seed),
                       "--output-root", str(output / name)]
            with (output / "logs" / f"{name}__{stage}.log").open("x") as log:
                result = subprocess.run(command, cwd=ROOT, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                stopped.set()
                return name, {"status": "failed", "stage": stage, "returncode": result.returncode,
                              "elapsed_seconds": time.perf_counter() - started}
        return name, {"status": "complete", "elapsed_seconds": time.perf_counter() - started}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, *job) for job in jobs]
        for future in as_completed(futures):
            name, result = future.result()
            record["jobs"][name] = result
            record["updated_utc"] = utc_now()
            write_json_atomic(output / "status.json", record)
            print(json.dumps({"job": name, **result}), flush=True)
    record["status"] = "complete" if all(j["status"] == "complete" for j in record["jobs"].values()) else "failed"
    record["finished_utc"] = utc_now()
    write_json_atomic(output / "status.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    outcome = launch(args.work_root, args.output_root, args.workers)
    print(json.dumps(outcome), flush=True)
    sys.exit(0 if outcome["status"] == "complete" else 1)
