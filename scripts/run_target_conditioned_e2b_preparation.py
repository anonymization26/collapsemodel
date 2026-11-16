#!/usr/bin/env python3
"""Run data preparation to completion, recording failure without starting experiments."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=ROOT)
    args = parser.parse_args()
    work = args.work_root.resolve()
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / "preparation.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another preparation pipeline is already running") from None
    status_path = work / "preparation_status.json"
    status = {"status": "running", "pid": os.getpid(),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "base_git_revision": os.environ.get("SOURCE_GIT_REVISION"),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "completed_steps": [], "real_outcome_experiments_started": False}

    def record():
        status["updated_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)

    env = {**os.environ, "E2B_WORK_ROOT": str(work), "E2B_PYTHON": sys.executable,
           "PYTHONDONTWRITEBYTECODE": "1"}
    commands = [(name, ["bash", str(ROOT / "scripts/prepare_target_conditioned_e2b_cuda.sh"), name])
                for name in ("check", "restore", "features", "bundles")]
    commands.append(("verify", [sys.executable, str(ROOT / "scripts/check_target_conditioned_e2b_preparation.py"),
                                "--work-root", str(work), "--output", str(work / "preparation_report.json")]))
    record()
    try:
        if not re.fullmatch(r"[0-9a-f]{40}", status["base_git_revision"] or ""):
            raise RuntimeError("SOURCE_GIT_REVISION must be a full base commit hash")
        for name, command in commands:
            status["step"] = name
            record()
            print(f"PREPARATION {name}", flush=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            status["completed_steps"].append(name)
        status["status"] = "ready"
    except BaseException as exc:
        status["status"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record()
        lock.close()


if __name__ == "__main__":
    main()
