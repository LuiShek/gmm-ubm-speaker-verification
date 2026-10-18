"""
GMM-UBM training, MAP adaptation, and likelihood-ratio scoring.

Implements the paper's Bayesian adaptation (Eq. 7-9):
    alpha_i = n_i / (n_i + r)
    mu_hat_i = alpha_i * E_i[x] + (1 - alpha_i) * mu_i
with relevance factor r = 16, means-only adaptation by default.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from . import config


# ---------------- log-domain GMM helpers ----------------

def _comp_logpdf(X: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    """(T, K) log N(x_t; mu_k, Sigma_k) for diagonal covariance."""
    T, D = X.shape
    prec = 1.0 / covars                       # (K, D)
    log_det = np.sum(np.log(2.0 * np.pi * covars), axis=1)   # (K,)
    out = np.empty((T, means.shape[0]), dtype=np.float64)
    # Vectorised over both frames and components: one broadcast per chunk
    # instead of a Python loop over K (ponytail: chunk trades RAM for speed;
    # 2000x256x38 float64 is ~156 MB, raised to 4x the frame chunk only when
    # free RAM allows - tune UBM_FRAME_CHUNK if you hit the 400 MB floor).
    step = max(1, config.UBM_FRAME_CHUNK)
    prec_t = prec.T                            # (D, K)
    means_t = means.T                          # (D, K)
    for s in range(0, T, step):
        Xc = X[s:s + step]                     # (t, D)
        # sum_d (x_d - mu_dk)^2 / var_dk  ==  sum_d x_d^2 p - 2 x_d mu p + mu^2 p
        q = ((Xc * Xc) @ prec_t
             - 2.0 * (Xc @ (means_t * prec_t))
             + np.sum(means_t * means_t * prec_t, axis=0)[None, :])
        out[s:s + step] = -0.5 * (log_det[None, :] + q)
    return out


def _log_joint(X, weights, means, covars):
    comp = _comp_logpdf(X, means, covars)
    return comp + np.log(np.maximum(weights, 1e-300))[None, :]


def frame_loglik(X, weights, means, covars) -> np.ndarray:
    """Per-frame log-likelihood under the GMM (marginalised over components)."""
    lj = _log_joint(X, weights, means, covars)
    return logsumexp(lj, axis=1)


def avg_loglik(X, weights, means, covars) -> float:
    """Paper Eq. 6 scaled by T (average log-likelihood)."""
    if X.shape[0] == 0:
        return float("-inf")
    return float(np.mean(frame_loglik(X, weights, means, covars)))


# ---------------- UBM training (EM) ----------------

def train_ubm(X: np.ndarray, n_mixtures: int = config.N_MIXTURES,
              max_iter: int = config.UBM_MAX_ITER, seed: int = config.SEED):
    """EM-train a diagonal-covariance GMM on pooled background frames."""
    rng = np.random.default_rng(seed)
    T, D = X.shape
    if T < n_mixtures:
        raise ValueError(f"Only {T} frames for {n_mixtures} mixtures")

    # k-means++ style init via scikit-learn for stability
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_mixtures, n_init=3, max_iter=100,
                random_state=seed).fit(X)
    means = km.cluster_centers_.astype(np.float64)
    weights = np.full(n_mixtures, 1.0 / n_mixtures, dtype=np.float64)
    covars = np.tile(np.var(X, axis=0) + config.COVARIANCE_FLOOR,
                     (n_mixtures, 1)).astype(np.float64)

    prev = -np.inf
    for it in range(1, max_iter + 1):
        # E-step: accumulate posteriors in chunks (RAM headroom)
        Nk = np.zeros(n_mixtures)
        Fk = np.zeros((n_mixtures, D))
        Sk = np.zeros((n_mixtures, D))
        total_ll = 0.0
        step = config.UBM_FRAME_CHUNK * 4
        for s in range(0, T, step):
            Xc = X[s:s + step]
            lj = _log_joint(Xc, weights, means, covars)
            ll = logsumexp(lj, axis=1)
            total_ll += float(np.sum(ll))
            post = np.exp(lj - ll[:, None])          # (t, K)
            Nk += post.sum(axis=0)
            Fk += post.T @ Xc
            Sk += post.T @ (Xc * Xc)

        # M-step
        weights = np.maximum(Nk / T, 1e-10)
        weights /= weights.sum()
        means = Fk / np.maximum(Nk[:, None], 1e-10)
        ex2 = Sk / np.maximum(Nk[:, None], 1e-10)
        covars = ex2 - means ** 2 + config.COVARIANCE_FLOOR

        avg = total_ll / T
        delta = avg - prev
        prev = avg
        print(f"  iter {it:3d}  avg loglik {avg:10.5f}  delta {delta:+.6f}", flush=True)
        if it > 3 and abs(delta) < 1e-6:
            print("  converged")
            break

    return {"weights": weights, "means": means, "covars": covars}


# ---------------- MAP adaptation ----------------

def map_adapt(ubm: dict, feats: list[np.ndarray],
              r: float = config.RELEVANCE_R) -> dict:
    """Paper Eq. 7-9. Means-only by default; r=16."""
    weights = ubm["weights"]
    means = ubm["means"].copy()
    covars = ubm["covars"].copy()

    X = np.concatenate([f for f in feats if f.shape[0] > 0], axis=0)
    if X.shape[0] == 0:
        return {"weights": weights.copy(), "means": means, "covars": covars}

    lj = _log_joint(X, weights, means, covars)
    ll = logsumexp(lj, axis=1)
    post = np.exp(lj - ll[:, None])
    Nk = post.sum(axis=0)
    Fk = post.T @ X
    Sk = post.T @ (X * X)

    alpha = Nk / (Nk + r)
    new_means = alpha[:, None] * (Fk / np.maximum(Nk[:, None], 1e-10)) \
        + (1.0 - alpha[:, None]) * means

    out = {"weights": weights.copy(), "means": new_means, "covars": covars}
    if config.ADAPT_VARS:
        ex2 = Sk / np.maximum(Nk[:, None], 1e-10)
        var = alpha[:, None] * ex2 + (1 - alpha[:, None]) * (covars + means ** 2) - new_means ** 2
        out["covars"] = var + config.COVARIANCE_FLOOR
    if config.ADAPT_WEIGHTS:
        w = alpha * (Nk / np.maximum(Nk.sum(), 1e-10)) + (1 - alpha) * weights
        out["weights"] = w / w.sum()
    return out


# ---------------- scoring ----------------

def ubm_loglik(X: np.ndarray, ubm: dict) -> float:
    """UBM average log-likelihood for a trial.

    Split out so callers can cache it: the UBM is fixed, so scoring N speaker
    models against the same utterance would otherwise recompute it N times.
    """
    if X.shape[0] == 0:
        return float("-inf")
    return avg_loglik(X, ubm["weights"], ubm["means"], ubm["covars"])


def score(model: dict, ubm: dict, X: np.ndarray, ubm_ll: float | None = None) -> float:
    """Average log-likelihood-ratio score (speaker - UBM).

    Pass ubm_ll from ubm_loglik() when scoring many models on the same
    utterance to avoid redundant work.
    """
    if X.shape[0] == 0:
        return float("-inf")
    s = avg_loglik(X, model["weights"], model["means"], model["covars"])
    u = ubm_ll if ubm_ll is not None else ubm_loglik(X, ubm)
    return s - u


# ---------------- persistence ----------------

def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)
