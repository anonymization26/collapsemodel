#!/usr/bin/env python3
"""Materialize a frozen E2b manifest from pinned DomainNet parquet shards."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable, Mapping

import fsspec
import pyarrow.parquet as pq
import requests


ROOT = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _read_manifest_rows(manifest_dir: Path) -> list[dict[str, str]]:
    manifest_path = manifest_dir / "manifest.json"
    samples_path = manifest_dir / "samples.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _sha256_file(samples_path) != manifest["samples_file_sha256"]:
        raise RuntimeError("samples.csv differs from the frozen manifest")
    with samples_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != int(manifest["sample_count"]):
        raise RuntimeError("manifest sample count mismatch")
    return rows


def _source_url(
    endpoint: str, repository: str, revision: str, path: str
) -> str:
    return (
        f"{endpoint.rstrip('/')}/datasets/{repository}/resolve/{revision}/{path}"
    )


def _remote_identity(url: str) -> tuple[int, str, str]:
    current = url
    for _ in range(4):
        response = requests.head(current, allow_redirects=False, timeout=(30, 60))
        if response.status_code in (301, 307, 308):
            current = response.headers["location"]
            continue
        response.raise_for_status()
        size_text = response.headers.get("x-linked-size")
        digest = response.headers.get("x-linked-etag", "").strip('"')
        revision = response.headers.get("x-repo-commit", "")
        if not size_text or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"source does not expose a pinned identity: {url}")
        return int(size_text), digest, revision
    raise RuntimeError(f"too many metadata redirects: {url}")


def _domain_names(parquet: pq.ParquetFile) -> list[str]:
    metadata = parquet.schema_arrow.metadata or {}
    payload = json.loads(metadata[b"huggingface"])
    return list(payload["info"]["features"]["domain"]["names"])


def _column_start(column: object) -> int:
    candidates = [
        value
        for value in (
            column.dictionary_page_offset,
            column.data_page_offset,
        )
        if value is not None and value >= 0
    ]
    if not candidates:
        raise RuntimeError("parquet column has no page offset")
    return min(candidates)


def _row_group_start(metadata: object, index: int) -> int:
    row_group = metadata.row_group(index)
    return min(
        _column_start(row_group.column(column_index))
        for column_index in range(row_group.num_columns)
    )


def _selected_row_groups(
    parquet: pq.ParquetFile, selected_domains: set[str]
) -> list[int]:
    names = _domain_names(parquet)
    selected_ids = {names.index(domain) for domain in selected_domains}
    metadata = parquet.metadata
    domain_index = next(
        index
        for index in range(metadata.num_columns)
        if metadata.schema.column(index).name == "domain"
    )
    selected = []
    for index in range(metadata.num_row_groups):
        statistics = metadata.row_group(index).column(domain_index).statistics
        if statistics is None or not statistics.has_min_max:
            raise RuntimeError("domain column lacks row-group statistics")
        endpoints = {int(statistics.min), int(statistics.max)}
        if endpoints & selected_ids:
            selected.append(index)
    return selected


def _runs(indices: Iterable[int]) -> list[tuple[int, int]]:
    values = sorted(set(indices))
    if not values:
        return []
    result = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            result.append((start, previous))
            start = value
        previous = value
    result.append((start, previous))
    return result


def _merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if start > end:
            raise RuntimeError("invalid byte range")
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _byte_ranges(
    parquet: pq.ParquetFile, file_size: int, row_groups: list[int]
) -> list[tuple[int, int]]:
    metadata = parquet.metadata
    footer_start = file_size - 8 - metadata.serialized_size
    if footer_start <= 4:
        raise RuntimeError("invalid parquet footer offset")
    starts = [
        _row_group_start(metadata, index)
        for index in range(metadata.num_row_groups)
    ]
    ranges = []
    for first, last in _runs(row_groups):
        start = 0 if first == 0 else starts[first]
        end = (
            starts[last + 1] - 1
            if last + 1 < metadata.num_row_groups
            else footer_start - 1
        )
        ranges.append((start, end))
    ranges.append((footer_start, file_size - 1))
    return _merge_ranges(ranges)


def _split_ranges(
    ranges: Iterable[tuple[int, int]], chunk_size: int
) -> list[tuple[int, int]]:
    chunks = []
    for start, end in ranges:
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + chunk_size - 1, end)
            chunks.append((cursor, chunk_end))
            cursor = chunk_end + 1
    return chunks


def _download_chunk(
    *, url: str, destination: Path, start: int, end: int, attempts: int = 8
) -> int:
    expected = end - start + 1
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                allow_redirects=True,
                stream=True,
                timeout=(30, 300),
            )
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError(f"range request returned {response.status_code}")
            observed_range = response.headers.get("content-range", "")
            if not observed_range.startswith(f"bytes {start}-{end}/"):
                raise RuntimeError(f"unexpected Content-Range: {observed_range}")
            descriptor = os.open(destination, os.O_RDWR)
            try:
                offset = start
                for payload in response.iter_content(chunk_size=1024 * 1024):
                    if payload:
                        view = memoryview(payload)
                        while view:
                            written = os.pwrite(descriptor, view, offset)
                            if written <= 0:
                                raise RuntimeError("pwrite made no progress")
                            offset += written
                            view = view[written:]
                if offset - start != expected:
                    raise RuntimeError(
                        f"short range response: expected={expected}, got={offset-start}"
                    )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return expected
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(min(30, 2**attempt))
    raise AssertionError("unreachable")


def _download_sparse(
    *,
    url: str,
    destination: Path,
    file_size: int,
    source_sha256: str,
    ranges: list[tuple[int, int]],
    workers: int,
    chunk_size: int,
) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    state_path = destination.with_suffix(destination.suffix + ".ranges.json")
    chunks = _split_ranges(ranges, chunk_size)
    identity = {
        "schema_version": "collapsemodel-sparse-parquet-v1",
        "source_sha256": source_sha256,
        "file_size": file_size,
        "ranges": [[start, end] for start, end in ranges],
        "chunks": [[start, end] for start, end in chunks],
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if {key: state[key] for key in identity} != identity:
            raise RuntimeError(f"download state differs for {destination.name}")
    else:
        state = {**identity, "completed": []}
        _write_json_atomic(state_path, state)
    if not destination.exists():
        with destination.open("wb") as stream:
            stream.truncate(file_size)
    elif destination.stat().st_size != file_size:
        raise RuntimeError(f"sparse file size differs for {destination.name}")

    completed = {tuple(item) for item in state["completed"]}
    pending = [chunk for chunk in chunks if chunk not in completed]
    downloaded = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_chunk,
                url=url,
                destination=destination,
                start=start,
                end=end,
            ): (start, end)
            for start, end in pending
        }
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            chunk = futures[future]
            downloaded += future.result()
            completed.add(chunk)
            state["completed"] = [list(item) for item in sorted(completed)]
            _write_json_atomic(state_path, state)
            if count == len(pending) or count % max(1, workers) == 0:
                print(
                    f"downloaded {destination.name}: "
                    f"{len(completed)}/{len(chunks)} chunks",
                    flush=True,
                )
    return {
        "path": destination.name,
        "source_sha256": source_sha256,
        "file_size": file_size,
        "downloaded_byte_count": sum(end - start + 1 for start, end in ranges),
        "row_group_ranges": [list(value) for value in ranges],
        "chunk_count": len(chunks),
        "new_bytes": downloaded,
    }


def _valid_output(path: Path, row: Mapping[str, str]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(row["byte_size"])
        and _sha256_file(path) == row["content_sha256"]
    )


def _reuse_existing(
    rows: list[dict[str, str]],
    dataset_root: Path,
    reuse_root: Path,
    reuse_domains: set[str],
) -> int:
    linked = 0
    for row in rows:
        if row["domain"] not in reuse_domains:
            continue
        output = dataset_root / row["relative_path"]
        if _valid_output(output, row):
            continue
        source = (
            reuse_root
            / row["official_split"]
            / row["class_name"]
            / f"{row['domain']}_{Path(row['archive_member']).name}"
        )
        if not _valid_output(source, row):
            raise RuntimeError(f"reused sample differs from manifest: {source}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        try:
            os.link(source, temporary)
        except OSError:
            _write_bytes_atomic(temporary, source.read_bytes())
        temporary.replace(output)
        linked += 1
    return linked


def _materialize_parquets(
    *,
    rows: list[dict[str, str]],
    dataset_root: Path,
    parquet_paths: list[tuple[Path, list[int]]],
    parquet_domains: set[str],
) -> int:
    wanted = {
        row["archive_member"]: row
        for row in rows
        if row["domain"] in parquet_domains
        and not _valid_output(dataset_root / row["relative_path"], row)
    }
    if len(wanted) != sum(
        1
        for row in rows
        if row["domain"] in parquet_domains
        and not _valid_output(dataset_root / row["relative_path"], row)
    ):
        raise RuntimeError("manifest archive paths are not unique")
    materialized = 0
    for parquet_path, row_groups in parquet_paths:
        parquet = pq.ParquetFile(parquet_path)
        for count, row_group_index in enumerate(row_groups, 1):
            table = parquet.read_row_group(
                row_group_index, columns=["image", "image_path"]
            )
            paths = table.column("image_path").to_pylist()
            images = table.column("image").to_pylist()
            for image_path, image in zip(paths, images):
                row = wanted.pop(image_path, None)
                if row is None:
                    continue
                payload = image["bytes"]
                if payload is None:
                    raise RuntimeError(f"parquet row has no image bytes: {image_path}")
                if len(payload) != int(row["byte_size"]):
                    raise RuntimeError(f"sample size mismatch: {image_path}")
                if hashlib.sha256(payload).hexdigest() != row["content_sha256"]:
                    raise RuntimeError(f"sample hash mismatch: {image_path}")
                _write_bytes_atomic(dataset_root / row["relative_path"], payload)
                materialized += 1
            if count % 100 == 0 or count == len(row_groups):
                print(
                    f"scanned {parquet_path.name}: {count}/{len(row_groups)} groups; "
                    f"remaining={len(wanted)}",
                    flush=True,
                )
    if wanted:
        raise RuntimeError(f"missing {len(wanted)} frozen samples in parquet sources")
    return materialized


def _verify_outputs(rows: list[dict[str, str]], dataset_root: Path) -> None:
    for index, row in enumerate(rows, 1):
        if not _valid_output(dataset_root / row["relative_path"], row):
            raise RuntimeError(f"materialized sample failed verification: {row['sample_id']}")
        if index % 5000 == 0 or index == len(rows):
            print(f"verified samples: {index}/{len(rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--source-config",
        type=Path,
        default=(
            ROOT
            / "code/configs/target_conditioned_e2b/domainnet_hf_parquet_v1.json"
        ),
    )
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--reuse-root", type=Path)
    parser.add_argument("--reuse-domains", nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=64)
    args = parser.parse_args()

    source = json.loads(args.source_config.read_text(encoding="utf-8"))
    rows = _read_manifest_rows(args.manifest_dir)
    all_domains = {row["domain"] for row in rows}
    reuse_domains = set(args.reuse_domains)
    if not reuse_domains <= all_domains:
        raise RuntimeError("unknown reuse domain")
    if reuse_domains and args.reuse_root is None:
        raise RuntimeError("--reuse-root is required with --reuse-domains")
    parquet_domains = all_domains - reuse_domains

    args.dataset_root.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    if reuse_domains:
        linked = _reuse_existing(
            rows, args.dataset_root, args.reuse_root, reuse_domains
        )
        print(f"reused frozen samples: {linked}", flush=True)

    sparse_reports = []
    parquet_paths: list[tuple[Path, list[int]]] = []
    for file_spec in source["files"]:
        url = _source_url(
            args.endpoint,
            source["repository"],
            source["revision"],
            file_spec["path"],
        )
        observed_size, observed_sha256, observed_revision = _remote_identity(url)
        if observed_size != int(file_spec["byte_size"]):
            raise RuntimeError(f"remote size changed: {file_spec['path']}")
        if observed_sha256 != file_spec["sha256"]:
            raise RuntimeError(f"remote SHA-256 changed: {file_spec['path']}")
        if observed_revision and observed_revision != source["revision"]:
            raise RuntimeError(f"remote revision changed: {file_spec['path']}")

        remote = pq.ParquetFile(
            fsspec.open(url, "rb", block_size=1024 * 1024).open()
        )
        row_groups = _selected_row_groups(remote, parquet_domains)
        ranges = _byte_ranges(remote, observed_size, row_groups)
        destination = args.work_dir / Path(file_spec["path"]).name
        report = _download_sparse(
            url=url,
            destination=destination,
            file_size=observed_size,
            source_sha256=observed_sha256,
            ranges=ranges,
            workers=args.workers,
            chunk_size=args.chunk_mib * 1024 * 1024,
        )
        report["source_path"] = file_spec["path"]
        report["selected_row_group_count"] = len(row_groups)
        sparse_reports.append(report)
        parquet_paths.append((destination, row_groups))

    materialized = _materialize_parquets(
        rows=rows,
        dataset_root=args.dataset_root,
        parquet_paths=parquet_paths,
        parquet_domains=parquet_domains,
    )
    _verify_outputs(rows, args.dataset_root)
    manifest = json.loads(
        (args.manifest_dir / "manifest.json").read_text(encoding="utf-8")
    )
    report = {
        "schema_version": "collapsemodel-e2b-materialization-v1",
        "manifest_id": manifest["manifest_id"],
        "source_repository": source["repository"],
        "source_revision": source["revision"],
        "source_config_sha256": _sha256_file(args.source_config),
        "reuse_domains": sorted(reuse_domains),
        "parquet_domains": sorted(parquet_domains),
        "reused_sample_count": linked,
        "parquet_materialized_sample_count": materialized,
        "verified_sample_count": len(rows),
        "sparse_sources": sparse_reports,
    }
    _write_json_atomic(args.work_dir / "materialization.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
