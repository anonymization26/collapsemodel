"""
Collapse Model
Transferability prediction baseline methods (for comparison experiments)

Implemented methods:
    - LogME:     Log Marginal Evidence (Zhang et al. 2021)
    - H-score:   Spectral measure of feature-label correlation (Bao et al. 2019)
    - TransRate: Coding-rate-based transferability estimation (Huang et al. 2022)

Used in E2 experiments for comparison with SA_k.

Reference implementations:
    LogME     -- github.com/thuml/LogME
    H-score   -- Algorithm 1 from original paper
    TransRate -- Appendix of original paper
"""

import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────
# 1. LogME（Log Marginal Evidence）
# ─────────────────────────────────────────────────────────────

class LogME:
    """
    LogME transferability score (Zhang et al., ICML 2021).

    Core idea: fit a Bayesian linear regression on features H and
    maximize the log marginal likelihood log p(y | H) of the labels.

    Higher LogME -> tighter linear relationship between features and labels -> better transferability.

    Usage:
        score = LogME().fit(H, y)
    """

    def __init__(self, regression: bool = False):
        """
        Args:
            regression: True for regression tasks, False for classification (per-class one-vs-rest)
        """
        self.regression = regression

    def fit(self, H: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the LogME score.

        Args:
            H: (N, d) feature matrix (L2-normalized)
            y: (N,) label array (integers for classification, floats for regression)

        Returns:
            LogME score (higher is better)
        """
        if self.regression:
            return self._fit_single(H, y.astype(np.float64))
        else:
            classes = np.unique(y)
            scores = []
            for c in classes:
                y_binary = (y == c).astype(np.float64)
                scores.append(self._fit_single(H, y_binary))
            return float(np.mean(scores))

    def _fit_single(self, H: np.ndarray, y: np.ndarray) -> float:
        """Single-target LogME computation (EM algorithm optimizing alpha, beta)."""
        N, d = H.shape
        H = H.astype(np.float64)
        y = y.astype(np.float64)

        # SVD of H
        U, s, Vt = np.linalg.svd(H, full_matrices=False)
        # s: (min(N,d),)

        # Initialize hyperparameters
        alpha = 1.0
        beta  = 1.0
        evidence = -np.inf

        for _ in range(100):  # EM iterations
            s2 = s ** 2

            # M step: update w = (alpha I + beta H^T H)^{-1} beta H^T y
            # Efficient computation via SVD
            Uy = U.T @ y       # (min_dim,)
            m  = beta * s * Uy / (alpha + beta * s2)   # (min_dim,) -- diagonal simplification

            # Compute log marginal likelihood
            # log p(y | H, alpha, beta) ~ -N/2 log(2pi) + N/2 log beta
            #   - beta/2 ||y - H m||^2 - alpha/2 ||m||^2
            #   + 1/2 log |alpha I + beta H^T H|^{-1}
            log_det = float(np.sum(np.log(alpha + beta * s2)))
            y_pred  = U @ (s * m)   # reconstruction of H w
            res_sq  = float(np.sum((y - y_pred) ** 2))
            new_evidence = (
                0.5 * d * np.log(alpha)
                + 0.5 * N * np.log(beta)
                - 0.5 * beta * res_sq
                - 0.5 * alpha * float(np.sum(m ** 2))
                - 0.5 * log_det
                - 0.5 * N * np.log(2 * np.pi)
            )

            # E step: update alpha, beta
            gamma = float(np.sum(beta * s2 / (alpha + beta * s2)))
            alpha = gamma / (float(np.sum(m ** 2)) + 1e-10)
            beta  = (N - gamma) / (res_sq + 1e-10)

            if abs(new_evidence - evidence) < 1e-5 * abs(new_evidence):
                evidence = new_evidence
                break
            evidence = new_evidence

        return float(evidence / N)   # normalized to be independent of dataset size


# ─────────────────────────────────────────────────────────────
# 2. H-score（Bao et al. 2019）
# ─────────────────────────────────────────────────────────────

def compute_hscore(H: np.ndarray, y: np.ndarray) -> float:
    """
    H-score transferability metric (Bao et al., AAAI 2019).

    Core idea:
        Ratio of between-class to within-class covariance:
        H-score = tr(Sigma_B^{-1} Sigma_W) (higher is better)

        where Sigma_B = between-class covariance, Sigma_W = within-class covariance (regularized)

    Args:
        H: (N, d) feature matrix (L2-normalized)
        y: (N,) integer labels

    Returns:
        H-score (higher -> better inter-class separation -> better transferability)
    """
    H = H.astype(np.float64)
    classes = np.unique(y)

    # Global mean
    mu_global = H.mean(axis=0)   # (d,)

    # Within-class covariance Sigma_W
    Sigma_W = np.zeros((H.shape[1], H.shape[1]))
    for c in classes:
        H_c = H[y == c]
        if len(H_c) < 2:
            continue
        mu_c = H_c.mean(axis=0)
        diff = H_c - mu_c[None, :]
        Sigma_W += diff.T @ diff

    n_total = len(H)
    Sigma_W = Sigma_W / n_total + 1e-4 * np.eye(H.shape[1])  # regularization

    # Between-class covariance Sigma_B
    Sigma_B = np.zeros_like(Sigma_W)
    for c in classes:
        H_c = H[y == c]
        n_c = len(H_c)
        mu_c = H_c.mean(axis=0)
        diff = (mu_c - mu_global)[:, None]
        Sigma_B += n_c * (diff @ diff.T)
    Sigma_B /= n_total

    # H-score = tr(Sigma_W^{-1} Sigma_B)
    try:
        Sigma_W_inv = np.linalg.inv(Sigma_W)
        score = float(np.trace(Sigma_W_inv @ Sigma_B))
    except np.linalg.LinAlgError:
        # Fall back to pseudoinverse if singular
        Sigma_W_inv = np.linalg.pinv(Sigma_W)
        score = float(np.trace(Sigma_W_inv @ Sigma_B))

    return score


# ─────────────────────────────────────────────────────────────
# 3. TransRate（Huang et al. 2022）
# ─────────────────────────────────────────────────────────────

def compute_transrate(
    H: np.ndarray,
    y: np.ndarray,
    eps: float = 0.5,
) -> float:
    """
    TransRate transferability score (Huang et al., ICML 2022).

    Based on the maximal coding rate difference:
        TransRate = R(Z) - R(Z | C)

    where R(Z) = overall coding rate, R(Z|C) = sum of per-class conditional coding rates.
    High TransRate -> globally spread but class-compact -> good transfer representation.

    Args:
        H:   (N, d) feature matrix
        y:   (N,) integer labels
        eps: coding precision (regularization parameter; 0.5 recommended by original paper)

    Returns:
        TransRate score (higher is better)
    """
    H = H.astype(np.float64)
    N, d = H.shape

    def coding_rate(Z: np.ndarray, eps_: float) -> float:
        """Coding rate R(Z, eps^2) = (d/2) log det(I + d/(N eps^2) Z Z^T)"""
        n, p = Z.shape
        C = (Z.T @ Z) / n
        _, s, _ = np.linalg.svd(C, full_matrices=False)
        # log det = Σ log(1 + p/(ε²) σᵢ)
        rate = 0.5 * float(np.sum(np.log(1.0 + p / (eps_ ** 2) * s)))
        return rate

    # Overall coding rate
    R_total = coding_rate(H, eps)

    # Conditional coding rate
    classes = np.unique(y)
    R_cond = 0.0
    for c in classes:
        H_c = H[y == c]
        n_c = len(H_c)
        if n_c < 2:
            continue
        R_c = coding_rate(H_c, eps)
        R_cond += (n_c / N) * R_c

    return float(R_total - R_cond)


# ─────────────────────────────────────────────────────────────
# 4. Linear Probe Accuracy（ground truth）
# ─────────────────────────────────────────────────────────────

def linear_probe_accuracy(
    H_train: np.ndarray,
    y_train: np.ndarray,
    H_test: np.ndarray,
    y_test: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
) -> float:
    """
    Linear probe accuracy (used as ground truth for E2 experiments).

    Uses sklearn LogisticRegression with L2 regularization.

    Args:
        H_train/H_test: (N, d) train/test features
        y_train/y_test: (N,) labels
        C:              inverse of regularization strength
        max_iter:       maximum number of iterations

    Returns:
        top-1 accuracy in [0, 1]
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    H_tr = scaler.fit_transform(H_train)
    H_te = scaler.transform(H_test)

    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs",
                              multi_class="multinomial", n_jobs=-1)
    clf.fit(H_tr, y_train)
    return float(clf.score(H_te, y_test))


# ─────────────────────────────────────────────────────────────
# 5. Unified interface
# ─────────────────────────────────────────────────────────────

def compute_all_baselines(
    H: np.ndarray,
    y: np.ndarray,
    H_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute all baseline transferability metrics in one call.

    Args:
        H:      (N, d) training features (L2-normalized)
        y:      (N,) labels
        H_test: test features (required for LP accuracy)
        y_test: test labels

    Returns:
        dict: {
            "logme":     float  LogME score
            "hscore":    float  H-score
            "transrate": float  TransRate
            "lp_acc":    float  Linear Probe accuracy (if test set provided)
        }
    """
    result = {}

    result["logme"]     = LogME().fit(H, y)
    result["hscore"]    = compute_hscore(H, y)
    result["transrate"] = compute_transrate(H, y)

    if H_test is not None and y_test is not None:
        result["lp_acc"] = linear_probe_accuracy(H, y, H_test, y_test)

    return result


# ─────────────────────────────────────────────────────────────
# 5. STG: Spectral Transfer Gap
#    Spectral-dimension metric for the dual-channel decomposition
# ─────────────────────────────────────────────────────────────

def compute_stg(H_src: np.ndarray, H_tgt: np.ndarray) -> float:
    """
    Spectral Transfer Gap (STG): Wasserstein-1 distance between the normalized
    singular-value distributions of source and target domains.

    Core idea:
        SVD decomposes H = U Sigma V^T into spectral (Sigma) and directional (V) channels.
        STG measures "how different the energy distributions are" (spectral channel),
        while SA_k measures "how different the activation directions are" (directional channel).
        The two are orthogonally complementary -- experiments show corr(STG, SA_k) ~= 0.3.

    Mathematical definition:
        p_src = sigma(H_src) / ||sigma(H_src)||_1     normalized singular-value distribution
        p_tgt = sigma(H_tgt) / ||sigma(H_tgt)||_1
        STG   = W_1(p_src, p_tgt) = ||CDF_src - CDF_tgt||_1

    Returns:
        float: STG >= 0; larger values indicate greater spectral distribution difference.

    Args:
        H_src: (N_src, d) source-domain feature matrix (L2-normalized)
        H_tgt: (N_tgt, d) target-domain feature matrix (L2-normalized)
    """
    # Compute singular values
    _, s_src, _ = np.linalg.svd(H_src, full_matrices=False)
    _, s_tgt, _ = np.linalg.svd(H_tgt, full_matrices=False)

    # Truncate to the common dimensionality (smaller singular-value count)
    k = min(len(s_src), len(s_tgt))
    s_src = s_src[:k]
    s_tgt = s_tgt[:k]

    # Normalize to probability distributions
    p_src = s_src / (s_src.sum() + 1e-12)
    p_tgt = s_tgt / (s_tgt.sum() + 1e-12)

    # Wasserstein-1 distance = L1 norm of CDF difference
    cdf_src = np.cumsum(p_src)
    cdf_tgt = np.cumsum(p_tgt)
    w1 = float(np.sum(np.abs(cdf_src - cdf_tgt)))

    return w1


def compute_dual_channel_distance(
    H_src: np.ndarray,
    H_tgt: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    k: int = 20,
) -> dict:
    """
    Dual-Channel Representation Distance.

    d_repr = alpha * STG + beta * (1 - SA_k)

    Returns:
        dict:
            "stg":    float   Spectral Transfer Gap
            "sa_k":   float   Subspace alignment (from metrics.subspace_alignment)
            "d_repr": float   Weighted dual-channel distance
            "d_dir":  float   Directional difference component = 1 - SA_k
    """
    from metrics.subspace_alignment import compute_subspace_alignment
    sa_result = compute_subspace_alignment(H_src, H_tgt, k=k)
    sa_k = sa_result["sa_k"]
    stg  = compute_stg(H_src, H_tgt)
    d_dir = 1.0 - sa_k
    d_repr = alpha * stg + beta * d_dir

    return {
        "stg":    stg,
        "sa_k":   sa_k,
        "d_dir":  d_dir,
        "d_repr": d_repr,
    }


# ─────────────────────────────────────────────────────────────
# 6. SVCCA（Raghu et al., NIPS 2017）
# ─────────────────────────────────────────────────────────────

def compute_svcca(H_A: np.ndarray, H_B: np.ndarray, k: int = 20) -> float:
    """
    SVCCA = (1/k) sum cos(theta_i)

    Arithmetic mean of principal cosines (no squaring, no weighting).

    Args:
        H_A: (N_A, d) feature matrix
        H_B: (N_B, d) feature matrix
        k:   subspace dimension

    Returns:
        SVCCA in [0, 1]
    """
    H_A = H_A.astype(np.float64)
    H_B = H_B.astype(np.float64)
    k_eff = max(1, min(k, min(H_A.shape), min(H_B.shape)))

    _, _, Vt_A = np.linalg.svd(H_A, full_matrices=False)
    _, _, Vt_B = np.linalg.svd(H_B, full_matrices=False)

    V_A = Vt_A[:k_eff].T  # (d, k_eff)
    V_B = Vt_B[:k_eff].T

    M = V_A.T @ V_B
    cos_theta = np.linalg.svd(M, compute_uv=False)
    cos_theta = np.clip(cos_theta, 0.0, 1.0)
    return float(np.mean(cos_theta))


# ─────────────────────────────────────────────────────────────
# 7. PWCCA（Morcos et al., NeurIPS 2018）
# ─────────────────────────────────────────────────────────────

def compute_pwcca(H_A: np.ndarray, H_B: np.ndarray, k: int = 20) -> float:
    """
    PWCCA = sum sigma_Ai^2 cos(theta_i) / sum sigma_Ai^2

    Uses squared source-domain SVD singular values as energy weights.

    Args:
        H_A: (N_A, d) source-domain feature matrix
        H_B: (N_B, d) target-domain feature matrix
        k:   subspace dimension

    Returns:
        PWCCA in [0, 1]
    """
    H_A = H_A.astype(np.float64)
    H_B = H_B.astype(np.float64)
    k_eff = max(1, min(k, min(H_A.shape), min(H_B.shape)))

    _, s_A, Vt_A = np.linalg.svd(H_A, full_matrices=False)
    _, _, Vt_B = np.linalg.svd(H_B, full_matrices=False)

    V_A = Vt_A[:k_eff].T
    V_B = Vt_B[:k_eff].T

    M = V_A.T @ V_B
    cos_theta = np.linalg.svd(M, compute_uv=False)
    cos_theta = np.clip(cos_theta, 0.0, 1.0)

    w = s_A[:k_eff] ** 2
    return float(np.dot(w, cos_theta) / (w.sum() + 1e-12))


# ─────────────────────────────────────────────────────────────
# 8. CKA（Linear Kernel, Kornblith et al. 2019）
# ─────────────────────────────────────────────────────────────

def compute_cka_linear(H_A: np.ndarray, H_B: np.ndarray) -> float:
    """
    Linear CKA = ||H_A^T H_B||_F^2 / (||H_A^T H_A||_F * ||H_B^T H_B||_F)

    Args:
        H_A: (N, d_A) feature matrix (N must be the same for both)
        H_B: (N, d_B) feature matrix

    Returns:
        CKA in [0, 1]
    """
    H_A = H_A.astype(np.float64)
    H_B = H_B.astype(np.float64)

    # Center the features
    H_A = H_A - H_A.mean(axis=0, keepdims=True)
    H_B = H_B - H_B.mean(axis=0, keepdims=True)

    cross = H_A.T @ H_B   # (d_A, d_B)
    norm_A = H_A.T @ H_A   # (d_A, d_A)
    norm_B = H_B.T @ H_B   # (d_B, d_B)

    hsic_ab = float(np.sum(cross ** 2))
    hsic_aa = float(np.sum(norm_A ** 2))
    hsic_bb = float(np.sum(norm_B ** 2))

    return hsic_ab / (np.sqrt(hsic_aa * hsic_bb) + 1e-12)


# ─────────────────────────────────────────────────────────────
# 9. PARC（Bolya et al., NeurIPS 2021）
# ─────────────────────────────────────────────────────────────

def compute_parc(H_A: np.ndarray, H_B: np.ndarray) -> float:
    """
    PARC = pearsonr(v_A, v_B)

    v_j = (1/N) sum_i H_{ij}^2  (per-dimension mean-squared activation profile)

    Args:
        H_A: (N_A, d) feature matrix
        H_B: (N_B, d) feature matrix (d must match)

    Returns:
        PARC in [-1, 1]
    """
    from scipy.stats import pearsonr
    H_A = H_A.astype(np.float64)
    H_B = H_B.astype(np.float64)
    v_A = np.mean(H_A ** 2, axis=0)
    v_B = np.mean(H_B ** 2, axis=0)
    r, _ = pearsonr(v_A, v_B)
    return float(r)


# ─────────────────────────────────────────────────────────────
# 10. UID (Two-NN intrinsic dimensionality estimator)
# ─────────────────────────────────────────────────────────────

def compute_uid_twonn(H: np.ndarray) -> float:
    """
    Two-NN intrinsic dimensionality estimator (Facco et al., Scientific Reports 2017).

    Computes the ratio of nearest-neighbor to second-nearest-neighbor distances
    on L2-normalized features, then estimates intrinsic dimension via extreme-value theory.

    Args:
        H: (N, d) feature matrix

    Returns:
        UID: estimated intrinsic dimension
    """
    H = H.astype(np.float64)
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    H_n = H / (norms + 1e-12)
    N = len(H_n)
    if N < 10:
        return np.nan

    # Euclidean distance (after normalization, monotonically related to spherical distance)
    dots = H_n @ H_n.T
    np.clip(dots, -1.0, 1.0, out=dots)
    dists = np.sqrt(np.maximum(2.0 - 2.0 * dots, 0.0))
    np.fill_diagonal(dists, np.inf)

    sorted_d = np.sort(dists, axis=1)
    mu1 = sorted_d[:, 0]
    mu2 = sorted_d[:, 1]
    valid = (mu1 > 1e-10) & (mu2 > 1e-10) & np.isfinite(mu1) & np.isfinite(mu2)
    if valid.sum() < 5:
        return np.nan
    ratio = mu2[valid] / mu1[valid]
    return float(1.0 / (np.log(2) * np.mean(np.log(ratio))))


# ─────────────────────────────────────────────────────────────
# 11. SSR（Spectral Specialization Ratio）
# ─────────────────────────────────────────────────────────────

def compute_ssr(
    H_ref: np.ndarray,
    H_tgt: np.ndarray,
    n_classes: int,
    k: int = 20,
) -> float:
    """
    SSR(M, T) = r_eff(H_{M,T}) / (SA_k(H_{M,ref}, H_{M,T}) * UID(H_{M,T}) * log(C_t))

    Args:
        H_ref:     (N_ref, d) reference dataset feature matrix
        H_tgt:     (N_tgt, d) target dataset feature matrix
        n_classes: C_t, number of classes in the target dataset
        k:         subspace dimension for SA_k

    Returns:
        SSR value (higher -> stronger spectral specialization)
    """
    from metrics.reff import effective_rank
    from metrics.subspace_alignment import compute_subspace_alignment

    r_eff = effective_rank(H_tgt)
    sa_result = compute_subspace_alignment(H_ref, H_tgt, k=k)
    sa_k = sa_result["sa_k"]
    uid = compute_uid_twonn(H_tgt)

    if not np.isfinite(uid) or uid < 1e-6:
        return np.nan
    if sa_k < 1e-6:
        return np.nan
    if n_classes < 2:
        return np.nan

    return r_eff / (sa_k * uid * np.log(n_classes))


# ─────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N, d, K = 300, 128, 5

    # Generate separable features (high transferability)
    H_good = np.zeros((N, d))
    y = np.repeat(np.arange(K), N // K)
    for c in range(K):
        H_good[y == c] = rng.standard_normal(d)[None, :] + 0.1 * rng.standard_normal(
            (N // K, d))
    H_good /= np.linalg.norm(H_good, axis=1, keepdims=True) + 1e-8

    # Generate non-separable features (low transferability)
    H_bad = rng.standard_normal((N, d))
    H_bad /= np.linalg.norm(H_bad, axis=1, keepdims=True) + 1e-8

    for label, H in [("Separable (high transfer)", H_good), ("Random (low transfer)", H_bad)]:
        r = compute_all_baselines(H, y)
        print(f"\n{label}:")
        for k, v in r.items():
            print(f"  {k:12s} = {v:.4f}")

    print("\nBaseline computation complete.")
