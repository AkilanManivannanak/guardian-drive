"""Analysis of self-recorded wearable ECG: windowing, signal quality and motion-artifact spectra."""
from __future__ import annotations

import re
from fractions import Fraction

import numpy as np
from scipy import signal as sps

from ..data.signal_utils import bandpass, detect_rpeaks, median_beat

FS, WIN, STEP = 100.0, 1000, 500
REST = re.compile(r"(still|rest|sit|baseline|calm)", re.I)


def is_reference(condition: str) -> bool:
    return bool(REST.search(condition))


def to_windows(x: np.ndarray, fs: float, trim_s: float = 1.0) -> np.ndarray:
    """Resample to 100 Hz, band-pass 0.5-40 Hz (as PTB-XL), drop edges, cut 10-s windows (5-s step)."""
    frac = Fraction(FS / fs).limit_denominator(1000)
    y = sps.resample_poly(np.nan_to_num(np.asarray(x, float)), frac.numerator, frac.denominator)
    y = bandpass(y, FS)
    k = int(trim_s * FS)
    y = y[k: len(y) - k] if len(y) > 2 * k + WIN else y
    if len(y) < WIN:
        return np.zeros((0, WIN), np.float32)
    starts = np.arange(0, len(y) - WIN + 1, STEP)
    w = np.stack([y[s: s + WIN] for s in starts])
    return (w - np.median(w, 1, keepdims=True)).astype(np.float32)


def beat_sqi(w: np.ndarray) -> float:
    """Mean correlation of each beat with the window's median beat (1 = identical beats); NaN if < 3 beats."""
    rp = detect_rpeaks(w, FS)
    pre, post = int(0.25 * FS), int(0.45 * FS)
    beats = [w[r - pre: r + post] for r in rp if r - pre >= 0 and r + post <= len(w)]
    if len(beats) < 3:
        return float("nan")
    B = np.stack(beats)
    t = np.median(B, 0)
    c = [np.corrcoef(b, t)[0, 1] for b in B]
    return float(np.nanmean(c))


def band_psd(w: np.ndarray, lo: float = 0.5, hi: float = 40.0):
    f, p = sps.welch(w, fs=FS, nperseg=256, axis=-1)
    m = (f >= lo) & (f <= hi)
    return f[m], p[..., m].mean(0) if p.ndim > 1 else p[m]


def excess_psd(motion: np.ndarray, still: np.ndarray):
    """Motion-artifact spectrum estimate: mean PSD of motion windows minus mean PSD of still windows."""
    f, pm = band_psd(motion)
    _, ps = band_psd(still)
    ex = np.maximum(pm - ps, 1e-4 * pm.max())
    snr_db = float(10 * np.log10(ps.sum() / ex.sum()))
    return f, ex, snr_db


def shape_match(target: np.ndarray, cand: np.ndarray) -> dict:
    """Compare spectral *shapes* (each normalised to unit band power): log-spectral distance (dB) and
    correlation of log-PSDs."""
    a = np.log10(target / target.sum())
    b = np.log10(cand / cand.sum())
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    return {"lsd_db": float(np.sqrt(np.mean((10 * (a - b)) ** 2))), "log_psd_corr": corr}
