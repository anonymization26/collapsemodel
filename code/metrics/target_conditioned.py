"""Target-conditioned dataset and sample selection primitives.

The primary objective is the target-weighted posterior variance

    J_T(S) = tr(C_T (Lambda_0 + sigma^-2 G_S)^-1),

under a frozen Bayesian linear model.  The implementation is NumPy-only and
keeps this risk objective separate from the D-optimal information-gain
baseline and from legacy effective-rank heuristics.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


_PSD_RELATIVE_TOLERANCE = 1e-10
_SELECTION_TIE_TOLERANCE = 1e-13


def _finite_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _symmetric_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = _finite_array(value, name)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square, got {matrix.shape}")
    scale = max(float(np.linalg.norm(matrix, ord=2)), 1.0)
    tolerance = _PSD_RELATIVE_TOLERANCE * scale
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
        error = float(np.max(np.abs(matrix - matrix.T)))
        raise ValueError(f"{name} is not symmetric; max error={error:.3e}")
    return (matrix + matrix.T) / 2.0


def _psd_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = _symmetric_matrix(value, name)
    if matrix.size == 0:
        return matrix
    minimum = float(np.linalg.eigvalsh(matrix)[0])
    tolerance = _PSD_RELATIVE_TOLERANCE * max(
        float(np.linalg.norm(matrix, ord=2)), 1.0
    )
    if minimum < -tolerance:
        raise ValueError(f"{name} is not PSD; lambda_min={minimum:.3e}")
    return matrix


def _pd_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = _symmetric_matrix(value, name)
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return matrix


def _inverse_pd(matrix: np.ndarray) -> np.ndarray:
    """Invert a validated positive-definite matrix through Cholesky solves."""

    chol = np.linalg.cholesky((matrix + matrix.T) / 2.0)
    inverse_chol = np.linalg.solve(chol, np.eye(matrix.shape[0], dtype=np.float64))
    inverse = inverse_chol.T @ inverse_chol
    return (inverse + inverse.T) / 2.0


def gram_from_features(
    features: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return ``sum_j weights[j] h_j h_j.T`` for row-wise features."""

    matrix = _finite_array(features, "features")
    if matrix.ndim != 2:
        raise ValueError(f"features must be two-dimensional, got {matrix.shape}")
    if weights is None:
        weighted = matrix
    else:
        sample_weights = _finite_array(weights, "weights")
        if sample_weights.shape != (matrix.shape[0],):
            raise ValueError(
                f"weights must have shape {(matrix.shape[0],)}, "
                f"got {sample_weights.shape}"
            )
        if np.any(sample_weights < 0.0):
            raise ValueError("weights must be nonnegative")
        weighted = np.sqrt(sample_weights)[:, None] * matrix
    gram = weighted.T @ weighted
    return (gram + gram.T) / 2.0


def second_moment(
    features: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate a feature second moment with optional nonnegative weights."""

    matrix = _finite_array(features, "features")
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("features must be a nonempty two-dimensional array")
    if weights is None:
        normalizer = float(matrix.shape[0])
    else:
        sample_weights = _finite_array(weights, "weights")
        if sample_weights.shape != (matrix.shape[0],):
            raise ValueError(
                f"weights must have shape {(matrix.shape[0],)}, "
                f"got {sample_weights.shape}"
            )
        normalizer = float(sample_weights.sum())
        if normalizer <= 0.0:
            raise ValueError("weights must have positive total mass")
    return gram_from_features(matrix, weights=weights) / normalizer


def posterior_covariance(
    prior_precision: np.ndarray,
    gram: Optional[np.ndarray] = None,
    noise_variance: float = 1.0,
) -> np.ndarray:
    """Compute ``(prior_precision + gram / noise_variance)^-1``."""

    prior = _pd_matrix(prior_precision, "prior_precision")
    if not math.isfinite(noise_variance) or noise_variance <= 0.0:
        raise ValueError("noise_variance must be finite and positive")
    if gram is None:
        information = prior
    else:
        contribution = _psd_matrix(gram, "gram")
        if contribution.shape != prior.shape:
            raise ValueError(
                f"gram shape {contribution.shape} does not match prior {prior.shape}"
            )
        information = prior + contribution / noise_variance
    return _inverse_pd(information)


def target_a_risk(target_moment: np.ndarray, posterior: np.ndarray) -> float:
    """Return the reducible target Bayes risk ``tr(C_T P)``."""

    target = _psd_matrix(target_moment, "target_moment")
    covariance = _pd_matrix(posterior, "posterior")
    if target.shape != covariance.shape:
        raise ValueError(
            f"target shape {target.shape} does not match posterior {covariance.shape}"
        )
    return _target_a_risk_validated(target, covariance)


def _target_a_risk_validated(target: np.ndarray, covariance: np.ndarray) -> float:
    value = float(np.trace(target @ covariance))
    tolerance = _PSD_RELATIVE_TOLERANCE * max(
        float(np.trace(target)) * float(np.trace(covariance)), 1.0
    )
    if value < -tolerance:
        raise FloatingPointError(f"target A-risk is negative: {value:.3e}")
    return max(value, 0.0)


def _target_a_objective_validated(
    target: np.ndarray,
    prior: np.ndarray,
    gram: np.ndarray,
    noise_variance: float,
) -> float:
    covariance = _inverse_pd(prior + gram / noise_variance)
    return _target_a_risk_validated(target, covariance)


def target_a_objective(
    target_moment: np.ndarray,
    prior_precision: np.ndarray,
    gram: Optional[np.ndarray] = None,
    noise_variance: float = 1.0,
) -> float:
    """Compute the target-weighted A-optimal objective for one design."""

    target = _psd_matrix(target_moment, "target_moment")
    prior = _pd_matrix(prior_precision, "prior_precision")
    if target.shape != prior.shape:
        raise ValueError("target_moment and prior_precision dimensions differ")
    if not math.isfinite(noise_variance) or noise_variance <= 0.0:
        raise ValueError("noise_variance must be finite and positive")
    contribution = (
        np.zeros_like(prior) if gram is None else _psd_matrix(gram, "gram")
    )
    if contribution.shape != prior.shape:
        raise ValueError("gram and prior_precision dimensions differ")
    return _target_a_objective_validated(
        target, prior, contribution, noise_variance
    )


def sample_marginal_gain(
    posterior: np.ndarray,
    target_moment: np.ndarray,
    feature: np.ndarray,
    noise_variance: float = 1.0,
    weight: float = 1.0,
) -> float:
    """Exact target-risk decrease from adding one weighted feature row."""

    covariance = _pd_matrix(posterior, "posterior")
    target = _psd_matrix(target_moment, "target_moment")
    vector = _finite_array(feature, "feature")
    if vector.shape != (covariance.shape[0],):
        raise ValueError(
            f"feature must have shape {(covariance.shape[0],)}, got {vector.shape}"
        )
    if target.shape != covariance.shape:
        raise ValueError("target_moment and posterior dimensions differ")
    if not math.isfinite(noise_variance) or noise_variance <= 0.0:
        raise ValueError("noise_variance must be finite and positive")
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("weight must be finite and nonnegative")
    projected = covariance @ vector
    numerator = weight * float(projected @ target @ projected)
    denominator = noise_variance + weight * float(vector @ projected)
    return max(numerator / denominator, 0.0)


def block_marginal_gain(
    posterior: np.ndarray,
    target_moment: np.ndarray,
    factor: np.ndarray,
    noise_variance: float = 1.0,
) -> float:
    """Exact target-risk decrease for a block with ``G = factor.T @ factor``."""

    covariance = _pd_matrix(posterior, "posterior")
    target = _psd_matrix(target_moment, "target_moment")
    rows = _finite_array(factor, "factor")
    if rows.ndim != 2 or rows.shape[1] != covariance.shape[0]:
        raise ValueError(
            f"factor must have shape (n, {covariance.shape[0]}), got {rows.shape}"
        )
    if target.shape != covariance.shape:
        raise ValueError("target_moment and posterior dimensions differ")
    if not math.isfinite(noise_variance) or noise_variance <= 0.0:
        raise ValueError("noise_variance must be finite and positive")
    if rows.shape[0] == 0:
        return 0.0
    projected = rows @ covariance
    middle = noise_variance * np.eye(rows.shape[0]) + projected @ rows.T
    solved = np.linalg.solve((middle + middle.T) / 2.0, projected)
    reduction = projected.T @ solved
    value = float(np.trace(target @ reduction))
    return max(value, 0.0)


def _prepare_blocks(
    blocks: Mapping[str, np.ndarray],
    dimension: int,
) -> dict[str, np.ndarray]:
    if not blocks:
        raise ValueError("blocks must be nonempty")
    prepared = {}
    for name in sorted(blocks):
        gram = _psd_matrix(blocks[name], f"blocks[{name!r}]")
        if gram.shape != (dimension, dimension):
            raise ValueError(
                f"blocks[{name!r}] has shape {gram.shape}, "
                f"expected {(dimension, dimension)}"
            )
        prepared[name] = gram
    return prepared


def aggregate_gram(
    blocks: Mapping[str, np.ndarray],
    selected: Iterable[str],
) -> np.ndarray:
    """Sum selected PSD blocks after checking names and dimensions."""

    names = list(selected)
    if not blocks:
        raise ValueError("blocks must be nonempty")
    first = _symmetric_matrix(next(iter(blocks.values())), "first block")
    prepared = _prepare_blocks(blocks, first.shape[0])
    unknown = sorted(set(names) - set(prepared))
    if unknown:
        raise KeyError(f"unknown block names: {unknown}")
    result = np.zeros_like(first)
    for name in names:
        result += prepared[name]
    return (result + result.T) / 2.0


def continuous_target_a(
    weights: np.ndarray,
    blocks: Sequence[np.ndarray],
    target_moment: np.ndarray,
    prior_precision: np.ndarray,
    noise_variance: float = 1.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return objective, gradient, and Hessian of the A-opt relaxation."""

    coefficients = _finite_array(weights, "weights")
    if coefficients.shape != (len(blocks),):
        raise ValueError(f"weights must have shape {(len(blocks),)}")
    if np.any(coefficients < 0.0):
        raise ValueError("weights must be nonnegative")
    if not math.isfinite(noise_variance) or noise_variance <= 0.0:
        raise ValueError("noise_variance must be finite and positive")
    prior = _pd_matrix(prior_precision, "prior_precision")
    target = _psd_matrix(target_moment, "target_moment")
    if target.shape != prior.shape:
        raise ValueError("target_moment and prior_precision dimensions differ")
    prepared = [
        _psd_matrix(block, f"blocks[{index}]") / noise_variance
        for index, block in enumerate(blocks)
    ]
    if any(block.shape != prior.shape for block in prepared):
        raise ValueError("all blocks must match prior_precision dimensions")
    information = prior.copy()
    for coefficient, block in zip(coefficients, prepared):
        information += coefficient * block
    covariance = _inverse_pd(information)
    objective = target_a_risk(target, covariance)
    gradient = np.empty(len(prepared), dtype=np.float64)
    hessian = np.empty((len(prepared), len(prepared)), dtype=np.float64)
    for index, block_i in enumerate(prepared):
        gradient[index] = -float(np.trace(target @ covariance @ block_i @ covariance))
        for column, block_j in enumerate(prepared):
            first = target @ covariance @ block_j @ covariance @ block_i @ covariance
            second = target @ covariance @ block_i @ covariance @ block_j @ covariance
            hessian[index, column] = float(np.trace(first + second))
    hessian = (hessian + hessian.T) / 2.0
    return objective, gradient, hessian


def d_opt_information_gain(
    prior_precision: np.ndarray,
    gram: Optional[np.ndarray] = None,
    noise_variance: float = 1.0,
) -> float:
    """Return Bayesian parameter information gain in nats."""

    prior = _pd_matrix(prior_precision, "prior_precision")
    if not math.isfinite(noise_variance) or noise_variance <= 0.0:
        raise ValueError("noise_variance must be finite and positive")
    if gram is None:
        information = prior
    else:
        contribution = _psd_matrix(gram, "gram")
        if contribution.shape != prior.shape:
            raise ValueError("gram and prior_precision dimensions differ")
        information = prior + contribution / noise_variance
    return _d_opt_information_gain_validated(prior, information)


def _d_opt_information_gain_validated(
    prior: np.ndarray,
    information: np.ndarray,
) -> float:
    sign_prior, logdet_prior = np.linalg.slogdet(prior)
    sign_information, logdet_information = np.linalg.slogdet(information)
    if sign_prior <= 0 or sign_information <= 0:
        raise FloatingPointError("positive-definite log determinant failed")
    return 0.5 * float(logdet_information - logdet_prior)


@dataclass(frozen=True)
class SelectionStep:
    iteration: int
    candidate: str
    objective_before: float
    objective_after: float
    marginal_gain: float
    cost: float
    gain_per_cost: float


@dataclass(frozen=True)
class SelectionResult:
    method: str
    selected: tuple[str, ...]
    objective: float
    total_cost: float
    steps: tuple[SelectionStep, ...]


def _prepare_costs(
    names: Sequence[str],
    costs: Optional[Mapping[str, float]],
) -> dict[str, float]:
    if costs is None:
        return {name: 1.0 for name in names}
    missing = sorted(set(names) - set(costs))
    extra = sorted(set(costs) - set(names))
    if missing or extra:
        raise ValueError(f"cost keys differ from blocks; missing={missing}, extra={extra}")
    result = {}
    for name in names:
        value = float(costs[name])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"cost for {name!r} must be finite and positive")
        result[name] = value
    return result


def greedy_target_a(
    blocks: Mapping[str, np.ndarray],
    target_moment: np.ndarray,
    prior_precision: np.ndarray,
    k: int,
    noise_variance: float = 1.0,
    costs: Optional[Mapping[str, float]] = None,
    budget: Optional[float] = None,
) -> SelectionResult:
    """Cost-aware deterministic greedy minimization of target A-risk."""

    prior = _pd_matrix(prior_precision, "prior_precision")
    target = _psd_matrix(target_moment, "target_moment")
    if not math.isfinite(noise_variance) or noise_variance <= 0.0:
        raise ValueError("noise_variance must be finite and positive")
    if target.shape != prior.shape:
        raise ValueError("target_moment and prior_precision dimensions differ")
    prepared = _prepare_blocks(blocks, prior.shape[0])
    names = sorted(prepared)
    if not isinstance(k, int) or k < 0 or k > len(names):
        raise ValueError(f"k must be in [0, {len(names)}]")
    item_costs = _prepare_costs(names, costs)
    maximum_cost = math.inf if budget is None else float(budget)
    if math.isnan(maximum_cost) or maximum_cost < 0.0:
        raise ValueError("budget must be nonnegative")

    selected = []
    steps = []
    total_cost = 0.0
    total_gram = np.zeros_like(prior)
    current = _target_a_objective_validated(
        target, prior, total_gram, noise_variance
    )
    for iteration in range(k):
        feasible = [
            name for name in names
            if name not in selected
            and total_cost + item_costs[name] <= maximum_cost + 1e-12
        ]
        if not feasible:
            break
        best_name = None
        best_after = math.inf
        best_ratio = -math.inf
        for name in feasible:
            after = _target_a_objective_validated(
                target, prior, total_gram + prepared[name], noise_variance
            )
            gain = max(current - after, 0.0)
            ratio = gain / item_costs[name]
            if ratio > best_ratio + _SELECTION_TIE_TOLERANCE:
                best_name = name
                best_after = after
                best_ratio = ratio
        assert best_name is not None
        gain = max(current - best_after, 0.0)
        steps.append(SelectionStep(
            iteration=iteration,
            candidate=best_name,
            objective_before=current,
            objective_after=best_after,
            marginal_gain=gain,
            cost=item_costs[best_name],
            gain_per_cost=gain / item_costs[best_name],
        ))
        selected.append(best_name)
        total_cost += item_costs[best_name]
        total_gram += prepared[best_name]
        current = best_after
    return SelectionResult(
        method="target_a_greedy",
        selected=tuple(selected),
        objective=current,
        total_cost=total_cost,
        steps=tuple(steps),
    )


def greedy_d_optimal(
    blocks: Mapping[str, np.ndarray],
    prior_precision: np.ndarray,
    k: int,
    noise_variance: float = 1.0,
    costs: Optional[Mapping[str, float]] = None,
    budget: Optional[float] = None,
) -> SelectionResult:
    """Cost-aware greedy maximization of Bayesian D-opt information gain."""

    prior = _pd_matrix(prior_precision, "prior_precision")
    if not math.isfinite(noise_variance) or noise_variance <= 0.0:
        raise ValueError("noise_variance must be finite and positive")
    prepared = _prepare_blocks(blocks, prior.shape[0])
    names = sorted(prepared)
    if not isinstance(k, int) or k < 0 or k > len(names):
        raise ValueError(f"k must be in [0, {len(names)}]")
    item_costs = _prepare_costs(names, costs)
    maximum_cost = math.inf if budget is None else float(budget)
    if math.isnan(maximum_cost) or maximum_cost < 0.0:
        raise ValueError("budget must be nonnegative")

    selected = []
    steps = []
    total_cost = 0.0
    total_gram = np.zeros_like(prior)
    current = _d_opt_information_gain_validated(prior, prior)
    for iteration in range(k):
        feasible = [
            name for name in names
            if name not in selected
            and total_cost + item_costs[name] <= maximum_cost + 1e-12
        ]
        if not feasible:
            break
        best_name = None
        best_after = -math.inf
        best_ratio = -math.inf
        for name in feasible:
            after = _d_opt_information_gain_validated(
                prior,
                prior + (total_gram + prepared[name]) / noise_variance,
            )
            gain = max(after - current, 0.0)
            ratio = gain / item_costs[name]
            if ratio > best_ratio + _SELECTION_TIE_TOLERANCE:
                best_name = name
                best_after = after
                best_ratio = ratio
        assert best_name is not None
        gain = max(best_after - current, 0.0)
        steps.append(SelectionStep(
            iteration=iteration,
            candidate=best_name,
            objective_before=current,
            objective_after=best_after,
            marginal_gain=gain,
            cost=item_costs[best_name],
            gain_per_cost=gain / item_costs[best_name],
        ))
        selected.append(best_name)
        total_cost += item_costs[best_name]
        total_gram += prepared[best_name]
        current = best_after
    return SelectionResult(
        method="bayesian_d_greedy",
        selected=tuple(selected),
        objective=current,
        total_cost=total_cost,
        steps=tuple(steps),
    )


def exhaustive_target_a(
    blocks: Mapping[str, np.ndarray],
    target_moment: np.ndarray,
    prior_precision: np.ndarray,
    k: int,
    noise_variance: float = 1.0,
    max_combinations: int = 100_000,
) -> SelectionResult:
    """Return the exact cardinality-``k`` A-optimum for a small pool."""

    prior = _pd_matrix(prior_precision, "prior_precision")
    target = _psd_matrix(target_moment, "target_moment")
    if not math.isfinite(noise_variance) or noise_variance <= 0.0:
        raise ValueError("noise_variance must be finite and positive")
    prepared = _prepare_blocks(blocks, prior.shape[0])
    names = sorted(prepared)
    if not isinstance(k, int) or k < 0 or k > len(names):
        raise ValueError(f"k must be in [0, {len(names)}]")
    count = math.comb(len(names), k)
    if count > max_combinations:
        raise ValueError(
            f"exhaustive search needs {count} combinations, "
            f"above max_combinations={max_combinations}"
        )
    best_names = None
    best_objective = math.inf
    for combination in itertools.combinations(names, k):
        gram = np.zeros_like(prior)
        for name in combination:
            gram += prepared[name]
        objective = _target_a_objective_validated(
            target, prior, gram, noise_variance
        )
        if objective < best_objective - _SELECTION_TIE_TOLERANCE:
            best_names = combination
            best_objective = objective
    assert best_names is not None
    return SelectionResult(
        method="target_a_exhaustive",
        selected=tuple(best_names),
        objective=best_objective,
        total_cost=float(k),
        steps=(),
    )


@dataclass(frozen=True)
class PSDLowRankSketch:
    """Truncated PSD factor and certified residual bounds."""

    factor: np.ndarray
    dimension: int
    tail_operator_bound: float
    tail_trace: float
    full_trace: float

    @property
    def rank(self) -> int:
        return int(self.factor.shape[0])

    @property
    def approximation(self) -> np.ndarray:
        return self.factor.T @ self.factor

    @property
    def storage_bytes(self) -> int:
        return int(self.factor.nbytes + 3 * np.dtype(np.float64).itemsize)


def make_psd_sketch(gram: np.ndarray, rank: int) -> PSDLowRankSketch:
    """Build an exact-eigendecomposition rank-``rank`` PSD sketch."""

    matrix = _psd_matrix(gram, "gram")
    dimension = matrix.shape[0]
    if not isinstance(rank, int) or rank < 0 or rank > dimension:
        raise ValueError(f"rank must be in [0, {dimension}]")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    retained = eigenvalues[:rank]
    factor = np.sqrt(retained)[:, None] * eigenvectors[:, :rank].T
    tail = eigenvalues[rank:]
    return PSDLowRankSketch(
        factor=factor,
        dimension=dimension,
        tail_operator_bound=float(tail[0]) if tail.size else 0.0,
        tail_trace=float(tail.sum()),
        full_trace=float(eigenvalues.sum()),
    )


def make_feature_sketch(
    features: np.ndarray,
    rank: int,
    weights: Optional[np.ndarray] = None,
) -> PSDLowRankSketch:
    """Build a PSD sketch directly from rows without materializing a full Gram."""

    matrix = _finite_array(features, "features")
    if matrix.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if weights is not None:
        sample_weights = _finite_array(weights, "weights")
        if sample_weights.shape != (matrix.shape[0],):
            raise ValueError("weights and feature rows differ")
        if np.any(sample_weights < 0.0):
            raise ValueError("weights must be nonnegative")
        matrix = np.sqrt(sample_weights)[:, None] * matrix
    maximum_rank = min(matrix.shape)
    if not isinstance(rank, int) or rank < 0 or rank > maximum_rank:
        raise ValueError(f"rank must be in [0, {maximum_rank}]")
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    retained = singular[:rank]
    factor = retained[:, None] * vh[:rank]
    tail_eigenvalues = singular[rank:] ** 2
    return PSDLowRankSketch(
        factor=factor,
        dimension=matrix.shape[1],
        tail_operator_bound=(
            float(tail_eigenvalues[0]) if tail_eigenvalues.size else 0.0
        ),
        tail_trace=float(tail_eigenvalues.sum()),
        full_trace=float(np.sum(singular ** 2)),
    )


@dataclass(frozen=True)
class RiskInterval:
    lower: float
    upper: float

    @property
    def width(self) -> float:
        return self.upper - self.lower


def sketch_target_a_interval(
    sketches: Iterable[PSDLowRankSketch],
    target_moment: np.ndarray,
    prior_precision: np.ndarray,
    noise_variance: float = 1.0,
) -> RiskInterval:
    """Return a deterministic A-risk interval for summed PSD sketches."""

    prior = _pd_matrix(prior_precision, "prior_precision")
    target = _psd_matrix(target_moment, "target_moment")
    if target.shape != prior.shape:
        raise ValueError("target_moment and prior_precision dimensions differ")
    approximation = np.zeros_like(prior)
    tail_bound = 0.0
    for sketch in sketches:
        if sketch.dimension != prior.shape[0]:
            raise ValueError("sketch and prior_precision dimensions differ")
        approximation += sketch.approximation
        tail_bound += sketch.tail_operator_bound
    upper = target_a_objective(
        target, prior, approximation, noise_variance
    )
    lower = target_a_objective(
        target,
        prior,
        approximation + tail_bound * np.eye(prior.shape[0]),
        noise_variance,
    )
    if lower > upper + _SELECTION_TIE_TOLERANCE:
        raise FloatingPointError(
            f"invalid risk interval: lower={lower:.6g}, upper={upper:.6g}"
        )
    return RiskInterval(lower=min(lower, upper), upper=max(lower, upper))


def candidate_sketch_intervals(
    selected: Sequence[str],
    candidates: Sequence[str],
    sketches: Mapping[str, PSDLowRankSketch],
    target_moment: np.ndarray,
    prior_precision: np.ndarray,
    noise_variance: float = 1.0,
) -> dict[str, RiskInterval]:
    """Compute next-step risk intervals for each remaining candidate."""

    unknown = sorted((set(selected) | set(candidates)) - set(sketches))
    if unknown:
        raise KeyError(f"unknown sketch names: {unknown}")
    prefix = [sketches[name] for name in selected]
    return {
        name: sketch_target_a_interval(
            prefix + [sketches[name]],
            target_moment,
            prior_precision,
            noise_variance,
        )
        for name in sorted(candidates)
    }


def certified_minimum(
    intervals: Mapping[str, RiskInterval],
    tolerance: float = 0.0,
) -> Optional[str]:
    """Return a uniquely certified minimizer, or ``None`` if intervals overlap."""

    if not intervals:
        raise ValueError("intervals must be nonempty")
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and nonnegative")
    names = sorted(intervals)
    if len(names) == 1:
        return names[0]
    certified = []
    for name in names:
        competitor_lower = min(
            intervals[other].lower for other in names if other != name
        )
        if intervals[name].upper + tolerance < competitor_lower:
            certified.append(name)
    if len(certified) > 1:
        raise FloatingPointError("more than one strict interval minimizer")
    return certified[0] if certified else None


@dataclass(frozen=True)
class AdaptiveSketchStep:
    iteration: int
    candidate: str
    certified: bool
    used_uncertified_fallback: bool
    refinement_rounds: int
    interval_lower: float
    interval_upper: float
    requested_ranks: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class AdaptiveSketchResult:
    selected: tuple[str, ...]
    objective: float
    transmitted_bytes: int
    full_gram_bytes: int
    steps: tuple[AdaptiveSketchStep, ...]

    @property
    def byte_ratio(self) -> float:
        return self.transmitted_bytes / self.full_gram_bytes

    @property
    def certificate_rate(self) -> float:
        if not self.steps:
            return 1.0
        return sum(step.certified for step in self.steps) / len(self.steps)


def adaptive_sketch_target_a(
    blocks: Mapping[str, np.ndarray],
    target_moment: np.ndarray,
    prior_precision: np.ndarray,
    k: int,
    rank_schedule: Sequence[int],
    noise_variance: float = 1.0,
    fallback_to_full: bool = True,
) -> AdaptiveSketchResult:
    """Greedily select blocks with certified, adaptively refined PSD sketches.

    At each step, only the shared selected prefix and candidates whose lower
    bounds overlap the best current upper bound are refined.  If requested,
    full rank is appended to the schedule so a unique exact minimizer is
    eventually certified; exact ties use the common lexicographic rule and are
    explicitly marked as a fallback.
    """

    prior = _pd_matrix(prior_precision, "prior_precision")
    target = _psd_matrix(target_moment, "target_moment")
    if target.shape != prior.shape:
        raise ValueError("target_moment and prior_precision dimensions differ")
    if not math.isfinite(noise_variance) or noise_variance <= 0.0:
        raise ValueError("noise_variance must be finite and positive")
    prepared = _prepare_blocks(blocks, prior.shape[0])
    names = sorted(prepared)
    if not isinstance(k, int) or k < 0 or k > len(names):
        raise ValueError(f"k must be in [0, {len(names)}]")
    schedule = sorted(set(int(rank) for rank in rank_schedule))
    if not schedule:
        raise ValueError("rank_schedule must be nonempty")
    if schedule[0] < 0 or schedule[-1] > prior.shape[0]:
        raise ValueError(f"rank_schedule must lie in [0, {prior.shape[0]}]")
    if fallback_to_full and schedule[-1] != prior.shape[0]:
        schedule.append(prior.shape[0])

    rank_positions = {name: 0 for name in names}
    sketch_cache: dict[tuple[str, int], PSDLowRankSketch] = {}

    def sketch_for(name: str) -> PSDLowRankSketch:
        rank = schedule[rank_positions[name]]
        key = (name, rank)
        if key not in sketch_cache:
            sketch_cache[key] = make_psd_sketch(prepared[name], rank)
        return sketch_cache[key]

    selected: list[str] = []
    steps: list[AdaptiveSketchStep] = []
    for iteration in range(k):
        candidates = [name for name in names if name not in selected]
        refinement_rounds = 0
        while True:
            current_sketches = {name: sketch_for(name) for name in names}
            intervals = candidate_sketch_intervals(
                selected,
                candidates,
                current_sketches,
                target,
                prior,
                noise_variance,
            )
            certificate = certified_minimum(intervals)
            if certificate is not None:
                choice = certificate
                certified = True
                used_uncertified_fallback = False
                break

            best_name = min(
                candidates, key=lambda name: (intervals[name].upper, name)
            )
            best_upper = intervals[best_name].upper
            contenders = [
                name for name in candidates
                if intervals[name].lower
                <= best_upper + _SELECTION_TIE_TOLERANCE
            ]
            refinable = sorted(
                name for name in set(selected) | set(contenders)
                if rank_positions[name] + 1 < len(schedule)
            )
            if not refinable:
                choice = best_name
                certified = False
                used_uncertified_fallback = True
                break
            for name in refinable:
                rank_positions[name] += 1
            refinement_rounds += 1

        chosen_interval = intervals[choice]
        selected.append(choice)
        steps.append(AdaptiveSketchStep(
            iteration=iteration,
            candidate=choice,
            certified=certified,
            used_uncertified_fallback=used_uncertified_fallback,
            refinement_rounds=refinement_rounds,
            interval_lower=chosen_interval.lower,
            interval_upper=chosen_interval.upper,
            requested_ranks=tuple(
                (name, schedule[rank_positions[name]]) for name in names
            ),
        ))

    transmitted_bytes = sum(sketch_for(name).storage_bytes for name in names)
    full_gram_bytes = len(names) * prior.shape[0] ** 2 * np.dtype(np.float64).itemsize
    selected_gram = sum(
        (prepared[name] for name in selected), start=np.zeros_like(prior)
    )
    objective = _target_a_objective_validated(
        target, prior, selected_gram, noise_variance
    )
    return AdaptiveSketchResult(
        selected=tuple(selected),
        objective=objective,
        transmitted_bytes=int(transmitted_bytes),
        full_gram_bytes=int(full_gram_bytes),
        steps=tuple(steps),
    )


@dataclass(frozen=True)
class RidgeStatistics:
    gram: np.ndarray
    cross: np.ndarray
    response_norm: float
    n_samples: int


def ridge_statistics(features: np.ndarray, responses: np.ndarray) -> RidgeStatistics:
    """Return additive sufficient statistics for squared-loss ridge."""

    matrix = _finite_array(features, "features")
    targets = _finite_array(responses, "responses")
    if matrix.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if targets.ndim == 1:
        targets = targets[:, None]
    if targets.ndim != 2 or targets.shape[0] != matrix.shape[0]:
        raise ValueError("responses must have one row per feature")
    return RidgeStatistics(
        gram=gram_from_features(matrix),
        cross=matrix.T @ targets,
        response_norm=float(np.sum(targets * targets)),
        n_samples=int(matrix.shape[0]),
    )


def combine_ridge_statistics(
    statistics: Iterable[RidgeStatistics],
) -> RidgeStatistics:
    """Add compatible ridge sufficient statistics."""

    items = list(statistics)
    if not items:
        raise ValueError("statistics must be nonempty")
    gram_shape = items[0].gram.shape
    cross_shape = items[0].cross.shape
    if any(item.gram.shape != gram_shape for item in items):
        raise ValueError("ridge Gram dimensions differ")
    if any(item.cross.shape != cross_shape for item in items):
        raise ValueError("ridge cross-statistic dimensions differ")
    return RidgeStatistics(
        gram=sum((item.gram for item in items), start=np.zeros(gram_shape)),
        cross=sum((item.cross for item in items), start=np.zeros(cross_shape)),
        response_norm=float(sum(item.response_norm for item in items)),
        n_samples=int(sum(item.n_samples for item in items)),
    )


def ridge_solution(statistics: RidgeStatistics, regularization: float) -> np.ndarray:
    """Solve ridge regression from sufficient statistics."""

    if not math.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("regularization must be finite and positive")
    information = _psd_matrix(statistics.gram, "statistics.gram")
    information = information + regularization * np.eye(information.shape[0])
    return np.linalg.solve(information, statistics.cross)


def ridge_squared_risk(
    train_statistics: RidgeStatistics,
    target_statistics: RidgeStatistics,
    regularization: float,
) -> float:
    """Evaluate exact target empirical squared risk from sufficient statistics."""

    weights = ridge_solution(train_statistics, regularization)
    if target_statistics.gram.shape != train_statistics.gram.shape:
        raise ValueError("train and target feature dimensions differ")
    if target_statistics.cross.shape[1] != weights.shape[1]:
        raise ValueError("train and target response dimensions differ")
    if target_statistics.n_samples <= 0:
        raise ValueError("target statistics must contain samples")
    loss = (
        float(np.trace(weights.T @ target_statistics.gram @ weights))
        - 2.0 * float(np.trace(weights.T @ target_statistics.cross))
        + target_statistics.response_norm
    ) / target_statistics.n_samples
    tolerance = _PSD_RELATIVE_TOLERANCE * max(target_statistics.response_norm, 1.0)
    if loss < -tolerance:
        raise FloatingPointError(f"ridge squared risk is negative: {loss:.3e}")
    return max(loss, 0.0)
