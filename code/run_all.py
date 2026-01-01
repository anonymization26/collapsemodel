#!/usr/bin/env python3
"""
Collapse Model: reproducibility orchestrator.

Reads `configs/repro_config.yaml`, runs each enabled experiment in order,
and reports which paper tables/figures the outputs feed.

Usage:
    python run_all.py                    # run all experiments marked enabled
    python run_all.py --dry-run          # list what would run, with timings
    python run_all.py --only e1a_synthetic,p1a_modern_baselines
    python run_all.py --skip e1d_clip_dino
    python run_all.py --config custom.yaml

Exit codes:
    0 = all enabled experiments completed (or skipped)
    1 = one or more experiments failed (continues running the rest)
    2 = configuration error (cannot start)
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(2)


REPRO_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = REPRO_ROOT / "scripts"


# ──────────────────────────────────────────────────────────────────
def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict) -> dict:
    p = cfg["paths"]
    p["repro_root"]    = Path(p.get("repro_root") or REPRO_ROOT).resolve()
    p["data_dir"]      = (p["repro_root"] / p["data_dir"]).resolve()    if not Path(p["data_dir"]).is_absolute() else Path(p["data_dir"])
    p["feature_cache"] = (p["repro_root"] / p["feature_cache"]).resolve() if not Path(p["feature_cache"]).is_absolute() else Path(p["feature_cache"])
    p["results_dir"]   = (p["repro_root"] / p["results_dir"]).resolve()   if not Path(p["results_dir"]).is_absolute() else Path(p["results_dir"])
    p["figures_dir"]   = (p["repro_root"] / p["figures_dir"]).resolve()   if not Path(p["figures_dir"]).is_absolute() else Path(p["figures_dir"])
    for k in ("data_dir", "feature_cache", "results_dir", "figures_dir"):
        p[k].mkdir(parents=True, exist_ok=True)
    return cfg


def select_experiments(cfg: dict, only: list, skip: list) -> list:
    """Return list of (name, exp_cfg) in dict-insertion order, after filtering."""
    out = []
    for name, exp in cfg["experiments"].items():
        if not exp.get("enabled", True):
            continue
        if only and name not in only:
            continue
        if name in skip:
            continue
        out.append((name, exp))
    return out


def run_one(name: str, exp: dict, env: dict, dry_run: bool, log_dir: Path) -> dict:
    """Run a single script. Return result record."""
    script = SCRIPTS_DIR / exp["script"]
    if not script.exists():
        return {"name": name, "status": "missing_script",
                "error": f"{script} not found"}

    cmd = [sys.executable, "-u", str(script)]
    pretty_cmd = " ".join(shlex.quote(c) for c in cmd)

    print(f"\n{'═' * 76}")
    print(f"  ▶ {name}  ({exp.get('expected_runtime_min', '?')} min expected)")
    print(f"  → {exp.get('description', '')}")
    print(f"  $ {pretty_cmd}")
    print('─' * 76, flush=True)

    if dry_run:
        return {"name": name, "status": "dry_run", "elapsed_s": 0.0}

    log_path = log_dir / f"{name}.log"
    t0 = time.time()
    with open(log_path, "w") as logf:
        logf.write(f"# {datetime.now().isoformat()}  $ {pretty_cmd}\n\n")
        logf.flush()
        try:
            r = subprocess.run(cmd, env=env, cwd=SCRIPTS_DIR,
                               stdout=logf, stderr=subprocess.STDOUT,
                               check=False)
            status = "ok" if r.returncode == 0 else f"failed (rc={r.returncode})"
        except KeyboardInterrupt:
            status = "interrupted"
        except Exception as e:
            status = f"exception: {e}"
    elapsed = time.time() - t0
    print(f"  ✔ {name}  {status}  ({elapsed:.0f}s)  → log: {log_path}",
          flush=True)
    return {"name": name, "status": status, "elapsed_s": elapsed,
            "log": str(log_path)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPRO_ROOT / "configs" / "repro_config.yaml"))
    ap.add_argument("--only", default="",
        help="Comma-separated experiment names to run (others skipped).")
    ap.add_argument("--skip", default="",
        help="Comma-separated experiment names to skip.")
    ap.add_argument("--dry-run", action="store_true",
        help="Print the plan without executing.")
    args = ap.parse_args()

    only = [s.strip() for s in args.only.split(",") if s.strip()]
    skip = [s.strip() for s in args.skip.split(",") if s.strip()]

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2

    cfg = resolve_paths(load_config(cfg_path))
    experiments = select_experiments(cfg, only, skip)

    if not experiments:
        print("No experiments selected (check --only / --skip and config).",
              file=sys.stderr)
        return 2

    # Compose env to forward to subprocesses
    env = os.environ.copy()
    env["COLLAPSE_REPRO_ROOT"] = str(cfg["paths"]["repro_root"])
    env["COLLAPSE_DATA_DIR"]   = str(cfg["paths"]["data_dir"])
    env["PYTHONPATH"]         = (
        f"{SCRIPTS_DIR}{os.pathsep}{REPRO_ROOT}"
        f"{os.pathsep}{env.get('PYTHONPATH', '')}"
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["RANDOM_SEED"] = str(cfg["compute"].get("random_seed", 42))

    log_dir = cfg["paths"]["results_dir"] / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight summary
    total_min = sum(e.get("expected_runtime_min", 0) for _, e in experiments)
    print(f"  Collapse Model :: reproduce")
    print(f"  Repro root  : {cfg['paths']['repro_root']}")
    print(f"  Data dir    : {cfg['paths']['data_dir']}")
    print(f"  Results dir : {cfg['paths']['results_dir']}")
    print(f"  Experiments : {len(experiments)}  (~{total_min} min total)")
    if args.dry_run:
        print(f"  --dry-run: plan only, no execution.")

    results = []
    for name, exp in experiments:
        results.append(run_one(name, exp, env, args.dry_run, log_dir))

    # Summary
    print(f"\n{'═' * 76}")
    print(f"  SUMMARY")
    print('═' * 76)
    fail = 0
    for r in results:
        mark = "✔" if r["status"] in ("ok", "dry_run") else "✘"
        if mark == "✘": fail += 1
        elapsed = f"{r.get('elapsed_s', 0):.0f}s" if r.get("elapsed_s") else "-"
        print(f"  {mark}  {r['name']:35s}  {r['status']:25s}  {elapsed}")

    if fail:
        print(f"\n  {fail}/{len(results)} experiments failed; check logs in {log_dir}/")
        return 1
    print(f"\n  All {len(results)} experiments passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
