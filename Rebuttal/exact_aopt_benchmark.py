"""Compare exact A-opt enumeration with additive lower-bound screening."""

from heapq import heappush, heapreplace
from statistics import median
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular


def _target_factor(moment):
    """Factor a possibly singular PSD target moment as factor @ factor.T."""
    values, vectors = np.linalg.eigh(moment)
    positive = values > 0
    return np.asfortranarray(vectors[:, positive] * np.sqrt(values[positive]))


def _score(parts, prior, target_factor):
    precision = prior.copy(order="F")
    for part in parts:
        precision += part
    lower = cholesky(precision, lower=True, overwrite_a=True, check_finite=False)
    whitened = solve_triangular(lower, target_factor, lower=True, check_finite=False)
    return float(np.linalg.norm(whitened, "fro") ** 2)


def _anchor_lower_bounds(blocks, candidates, prior, moment, factor, anchor_name):
    block_multiplicity = np.bincount(
        candidates.ravel(), minlength=len(blocks)
    ).astype(float) / len(candidates)
    mean_gram = np.einsum("i,ijk->jk", block_multiplicity, blocks)
    if anchor_name == "mean_precision":
        anchor = prior + mean_gram
    elif anchor_name == "moment_match":
        # Under equal-cost, unit-normalized rows the mean trace equals the row count.
        anchor = prior + float(np.trace(mean_gram)) * moment
    else:
        raise ValueError(f"Unknown anchor {anchor_name}")
    anchor_factor = cho_factor(anchor, lower=True, check_finite=False)
    projected = cho_solve(anchor_factor, factor, check_finite=False)
    weight = projected @ projected.T
    value_at_anchor = float(np.sum(factor * projected))
    intercept = value_at_anchor + float(np.sum(weight * (anchor - prior)))
    block_terms = np.einsum("ij,kij->k", weight, blocks)
    lower_bounds = intercept - block_terms[candidates].sum(axis=1)
    # Retain near-boundary candidates using an absolute floor and a scale for
    # the subtraction. This is a float64 guard, not a validated error interval.
    scale = max(1.0, abs(intercept) + float(np.abs(block_terms[candidates]).sum(axis=1).max()))
    return lower_bounds, 1e-10 * scale


def _bounded_search(blocks, candidates, prior, moment, factor, budget, anchor_name):
    bounds, guard = _anchor_lower_bounds(blocks, candidates, prior, moment, factor, anchor_name)
    heap = []
    solves = 0
    # Candidate indices follow lexicographic ID order, so stable sorting
    # preserves the deterministic secondary tie breaker.
    for index in np.argsort(bounds, kind="stable"):
        if len(heap) == budget and bounds[index] > -heap[0][0] + guard:
            break
        score = _score(blocks[candidates[index]], prior, factor)
        solves += 1
        item = (-score, -int(index))
        if len(heap) < budget:
            heappush(heap, item)
        elif item > heap[0]:
            heapreplace(heap, item)
    return [-item[1] for item in sorted(heap, reverse=True)], solves


def benchmark(grams, target_moment, combos, budgets=(50, 227), repeats=3):
    """Time exact and bounded top-k screening with the unit-precision prior.

    Times include target factorization and all work from pool Grams to ranked
    candidates. Feature extraction and Gram formation are shared upstream work.
    A separate bound scan is timed for each budget. Exhaustive timings include
    scoring every candidate and sorting the resulting scores once.
    """
    names = sorted(grams)
    block_index = {name: i for i, name in enumerate(names)}
    blocks = np.stack([grams[name] for name in names]).astype(np.float64, copy=False)
    blocks = (blocks + blocks.transpose(0, 2, 1)) / 2
    moment = np.asarray(target_moment, dtype=np.float64)
    moment = (moment + moment.T) / 2
    dimension = moment.shape[0]
    prior = np.eye(dimension)
    ids = sorted(combos)
    candidates = np.array([[block_index[name] for name in combos[key]] for key in ids], dtype=int)
    budgets = tuple(sorted(set(budgets)))
    if not ids or not budgets or min(budgets) < 1 or max(budgets) > len(ids) or repeats < 1:
        raise ValueError("Expected nonempty candidates, valid budgets, and at least one repeat")

    exhaustive_times = []
    reference = None
    for _ in range(repeats):
        start = perf_counter()
        factor = _target_factor(moment)
        scores = [_score(blocks[parts], prior, factor) for parts in candidates]
        ranked = np.argsort(scores, kind="stable").tolist()
        exhaustive_times.append(perf_counter() - start)
        if reference is None:
            reference = ranked
        elif ranked != reference:
            raise RuntimeError("Exhaustive order changed between repetitions")

    result = {
        "candidate_count": len(ids),
        "dimension": dimension,
        "prior_precision": 1.0,
        "exhaustive": {
            "median_seconds": median(exhaustive_times),
            "seconds": exhaustive_times,
            "solve_count": len(ids),
            "top_k": {str(k): [ids[index] for index in reference[:k]] for k in budgets},
        },
        "anchors": {},
    }
    for anchor in ("mean_precision", "moment_match"):
        by_budget = {}
        for budget in budgets:
            times, solve_counts, matches = [], [], []
            for _ in range(repeats):
                start = perf_counter()
                factor = _target_factor(moment)
                ranked, solves = _bounded_search(blocks, candidates, prior, moment, factor, budget, anchor)
                times.append(perf_counter() - start)
                solve_counts.append(solves)
                matches.append(ranked == reference[:budget])
            by_budget[str(budget)] = {
                "median_seconds": median(times),
                "seconds": times,
                "solve_count": solve_counts[0],
                "solve_counts": solve_counts,
                "top_k": [ids[index] for index in ranked],
                "matches_exhaustive": all(matches),
            }
        result["anchors"][anchor] = by_budget
    return result
