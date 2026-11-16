"""Combine six-domain transport replay Brier and same-run screening time."""

import argparse
import csv
import hashlib
import json
from math import isfinite
from pathlib import Path
from statistics import mean

from replay_transport import BASELINES, BUDGETS, DOMAINS, ENCODERS, TRANSPORT


METHODS = (*TRANSPORT, *BASELINES, "random")
DISPLAY = {
    "sliced_w2": "Sliced Wasserstein",
    "whitened_sliced_w2": "Whitened sliced Wasserstein",
    "target_a": "A-opt",
    "second_moment_mmd": "MMD",
    "random": "Random (20 orders)",
    "full": "Full validation",
}


def read_timings(root):
    metadata = json.loads((root / "replay/screen_metadata.json").read_text())
    hashes = {(row["encoder"], row["domain"]): row["screen_sha256"] for row in metadata}
    expected = {(encoder, domain) for encoder in ENCODERS for domain in DOMAINS}
    if set(hashes) != expected or len(metadata) != len(expected):
        raise ValueError("Expected replay metadata for all 12 encoder/domain units")

    timings = {}
    for encoder, domain in sorted(expected):
        path = root / "units" / encoder / domain / "screen.json"
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != hashes[encoder, domain]:
            raise ValueError(f"Screen no longer matches replay metadata: {path}")
        screen = json.loads(payload)
        if (screen["encoder"], screen["domain"]) != (encoder, domain):
            raise ValueError(f"Incorrect encoder/domain metadata in {path}")
        for method in (*TRANSPORT, *BASELINES):
            if method not in screen["timing"]:
                continue
            duration = float(screen["timing"][method]["median_seconds"])
            if not isfinite(duration) or duration <= 0:
                raise ValueError(f"Invalid duration for {method} in {path}")
            timings[encoder, domain, method] = duration

    for method in (*TRANSPORT, *BASELINES):
        count = sum(key[-1] == method for key in timings)
        if count != len(expected) and (count or method in TRANSPORT):
            raise ValueError(f"Incomplete screening times for {method}: {count}/12")
    return timings


def read_brier(path, keys, expected):
    with path.open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    rows = {
        tuple(int(row[key]) if key == "budget" else row[key] for key in keys): row
        for row in raw
    }
    if set(rows) != expected or len(rows) != len(raw):
        raise ValueError(f"Missing or duplicate rows in {path}")
    for row in raw:
        if not all(isfinite(float(row[field])) for field in
                   ("test_brier", "full_test_brier", "delta_brier")):
            raise ValueError(f"Nonfinite Brier value in {path}")
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(rows, scope):
    extra_header = ["Domain"] if scope == "domain" else []
    extra_end = ["Domains above +0.005"] if scope == "overall" else []
    header = extra_header + ["Budget", "Screen", "Test Brier", "Gap to full", "Screen seconds"] + extra_end
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        fields = [row["domain"]] if scope == "domain" else []
        duration = row["screen_seconds"]
        fields += [row["budget"], DISPLAY[row["method"]],
                   f"{float(row['test_brier']):.6f}", f"{float(row['delta_brier']):+.6f}",
                   f"{duration:.4f}" if duration is not None else "—"]
        if scope == "overall":
            fields.append(f"{row['domains_over_005']}/{row['domain_count']}")
        lines.append("| " + " | ".join(fields) + " |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    timings = read_timings(root)
    expected = {(budget, method) for budget in BUDGETS for method in METHODS} | {(455, "full")}
    domain = read_brier(root / "replay/per_domain.csv", ("domain", "budget", "method"),
                        {(domain, *key) for domain in DOMAINS for key in expected})
    overall = read_brier(root / "replay/overall.csv", ("budget", "method"), expected)

    domain_times = {}
    for (encoder, target, method), duration in sorted(timings.items()):
        domain_times.setdefault((target, method), []).append(duration)
    domain_means = {key: mean(durations) for key, durations in domain_times.items()}
    overall_times = {
        method: mean(domain_means[target, method] for target in DOMAINS)
        for method in (*TRANSPORT, *BASELINES) if (DOMAINS[0], method) in domain_means
    }

    key_order = [(budget, method) for budget in BUDGETS for method in METHODS] + [(455, "full")]
    domain_rows = [
        {**domain[target, budget, method], "screen_seconds": domain_means.get((target, method))}
        for target in DOMAINS for budget, method in key_order
    ]
    overall_rows = [
        {**overall[budget, method], "screen_seconds": overall_times.get(method)}
        for budget, method in key_order
    ]
    timing_rows = [
        dict(encoder=encoder, domain=target, method=method, median_seconds=duration)
        for (encoder, target, method), duration in sorted(timings.items())
    ]
    write_csv(root / "timing_per_encoder_domain.csv", timing_rows)
    write_csv(root / "results_per_domain.csv", domain_rows)
    write_csv(root / "results_overall.csv", overall_rows)

    lines = [
        "# DomainNet six-domain fixed-pool transport comparison", "",
        "Six target domains and two encoders at frozen projection seed 20260911. Each shortlist "
        "winner minimizes archived logistic validation Brier. The signed gap subtracts the "
        "test Brier of the winner selected by validation over all 455 candidates. Lower Brier is "
        "better. Random first averages 20 archived orders per encoder and domain.", "",
        "Screening times are per-unit medians from the server runs, averaged over the two encoders "
        "within each domain and equally over domains. A ranking is computed once and serves both "
        "budgets, so its time is the same in both rows. Times use resident projected features and "
        "exclude image extraction, readout training, and validation. Missing timings are shown "
        "as em dashes. The transport variants use frozen features under the same pool constraints.", "",
        "## Overall", "", *table(overall_rows, "overall"), "",
        "## Per domain", "", *table(domain_rows, "domain"), "",
        "Per-encoder medians are in timing_per_encoder_domain.csv. Run-level screen timings and "
        "archive provenance are recorded in replay/screen_metadata.json.",
    ]
    (root / "Results.md").write_text("\n".join(lines) + "\n")
    print(f"PASS: matched 12 replay screen hashes; wrote {len(domain_rows)} domain rows, "
          f"{len(overall_rows)} overall rows and {len(timing_rows)} timing rows.")


if __name__ == "__main__":
    main()
