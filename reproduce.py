"""Verify, unpack, or recompute the recorded analyses without training models."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
BASE = Path('results/target_conditioned')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_distribution():
    manifest = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text())
    for relative, expected in manifest['files'].items():
        file = ROOT / relative
        if not file.is_file() or file.is_symlink() or digest(file) != expected:
            raise ValueError(f'Missing or changed distributed file: {relative}')
    print(f"PASS: {len(manifest['files'])} distributed file hashes")


def records(unpack=False):
    entries = json.loads((ROOT / 'data/records_manifest.json').read_text())['files']
    expected = {r['path']: r for r in entries}
    if len(expected) != len(entries):
        raise ValueError('Duplicate archived record')
    seen = set()
    with tarfile.open(ROOT / 'data/experiment_records.tar.gz', 'r:gz') as archive:
        for member in archive:
            relative = Path(member.name)
            if (not member.isfile() or relative.is_absolute() or '..' in relative.parts
                    or not member.name.startswith('results/') or member.name not in expected
                    or member.name in seen):
                raise ValueError(f'Unsafe or unlisted archive member: {member.name}')
            entry = expected[member.name]
            if member.size != entry['bytes']:
                raise ValueError(f'Incorrect record size: {member.name}')
            with archive.extractfile(member) as stream:
                data = stream.read()
            if hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise ValueError(f'Changed archived record: {member.name}')
            if unpack:
                target = ROOT / relative
                if not target.resolve().is_relative_to(ROOT):
                    raise ValueError(f'Output escapes package: {member.name}')
                if target.is_symlink():
                    raise ValueError(f'Refusing symlink: {member.name}')
                if target.exists():
                    if not target.is_file() or digest(target) != entry['sha256']:
                        raise ValueError(f'Refusing to overwrite changed record: {member.name}')
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('xb') as stream:
                        stream.write(data)
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('Missing records in archive')
    print(f"PASS: {len(seen)} records {'unpacked' if unpack else 'verified'} without changing their bytes")


def statistics(output):
    if not (ROOT / BASE).is_dir():
        raise ValueError('Run python3 reproduce.py unpack first')
    output.mkdir(parents=True, exist_ok=False)
    readout = BASE / 'revision_20260917/revision_results/e2b_readout_projection_v1'
    decision = BASE / 'revision_20260922/e2b_decision_cost_v1'
    commands = [
        ('primary', ['summarize_target_conditioned_e2b.py', '--config', 'code/configs/target_conditioned_e2b/domainnet_v1.json',
                     '--audit-dir', str(BASE / 'e2b_fixed_cost_shortlist/domainnet_v1/test_audit/resnet50'),
                     '--audit-dir', str(BASE / 'e2b_fixed_cost_shortlist/domainnet_v1/test_audit/dinov2_b14'),
                     '--output-dir', str(output / 'primary')]),
        ('readout', ['summarize_e2b_readout_projection.py', '--input-root', str(readout), '--output-root', str(output / 'readout')]),
        ('budget', ['analyze_e2b_decision_budget.py', '--input-root', str(readout),
                    '--protocol', 'code/configs/target_conditioned_e2b/decision_cost_v1.json', '--output-root', str(output / 'budget')]),
        ('challenge', ['summarize_e2b_decision_challenge.py', '--input-root', str(decision), '--output-root', str(output / 'challenge')]),
        ('domain_ties', ['analyze_e2b_domain_ties.py', '--input-root', str(decision),
                         '--verified-summary', str(output / 'challenge/summary.json'), '--output-root', str(output / 'domain_ties')]),
        ('cost', ['summarize_e2b_decision_cost.py', '--input-root', str(decision), '--output-root', str(output / 'cost')]),
    ]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for name, args in commands:
        args[0] = str(Path('scripts') / args[0])
        with (output / f'{name}.log').open('w') as log:
            result = subprocess.run([sys.executable, *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {output / (name + ".log")}')
        print(f'PASS: recomputed {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'unpack', 'stats'])
    parser.add_argument('--output', type=Path, default=ROOT / 'recomputed')
    args = parser.parse_args()
    if args.action == 'stats':
        statistics(args.output.resolve())
    else:
        verify_distribution()
        records(unpack=args.action == 'unpack')


if __name__ == '__main__':
    main()
