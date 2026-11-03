"""Distribution-level fidelity metrics for generated / simulated time series."""
from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.stats import wasserstein_distance

from ..data.signal_utils import heart_rate, welch_logpsd


def frechet_distance(fa: np.ndarray, fb: np.ndarray) -> float:
    """Frechet distance between Gaussians fitted to two feature sets (FID-style)."""
    mu_a, mu_b = fa.mean(0), fb.mean(0)
    ca, cb = np.cov(fa, rowvar=False), np.cov(fb, rowvar=False)
    covmean = linalg.sqrtm(ca @ cb)
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(ca + cb - 2 * covmean))


def mmd_rbf(a: np.ndarray, b: np.ndarray, bandwidth: float | None = None, max_n: int = 2000,
            seed: int = 0) -> float:
    """Unbiased squared MMD with a Gaussian kernel (median-heuristic bandwidth on the pooled set)."""
    rng = np.random.default_rng(seed)
    if len(a) > max_n:
        a = a[rng.choice(len(a), max_n, replace=False)]
    if len(b) > max_n:
        b = b[rng.choice(len(b), max_n, replace=False)]
    z = np.concatenate([a, b]).astype(np.float64)
    sq = (z**2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None] - 2 * z @ z.T, 0)
    if bandwidth is None:
        bandwidth = np.sqrt(0.5 * np.median(d2[np.triu_indices_from(d2, 1)]))
    k = np.exp(-d2 / (2 * bandwidth**2))
    n, m = len(a), len(b)
    kaa, kbb, kab = k[:n, :n], k[n:, n:], k[:n, n:]
    return float((kaa.sum() - np.trace(kaa)) / (n * (n - 1)) + (kbb.sum() - np.trace(kbb)) / (m * (m - 1))
                 - 2 * kab.mean())


def log_spectral_distance(a: np.ndarray, b: np.ndarray, fs: float = 100.0) -> float:
    """Mean absolute difference of the average Welch PSDs, in dB, over 0.5-40 Hz."""
    f, pa = welch_logpsd(a, fs)
    _, pb = welch_logpsd(b, fs)
    band = (f >= 0.5) & (f <= 40)
    return float(np.mean(np.abs(10 * pa.mean(0)[band] - 10 * pb.mean(0)[band])))


def hr_stats(x: np.ndarray, fs: float = 100.0) -> np.ndarray:
    return np.array([heart_rate(r, fs)[0] for r in x], dtype=np.float32)


def hr_wasserstein(a: np.ndarray, b: np.ndarray, fs: float = 100.0) -> float:
    ha, hb = hr_stats(a, fs), hr_stats(b, fs)
    ha, hb = ha[np.isfinite(ha)], hb[np.isfinite(hb)]
    return float(wasserstein_distance(ha, hb))
