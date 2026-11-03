"""Classical signal-processing utilities (numpy / scipy) shared by data prep and evaluation."""
from __future__ import annotations

import numpy as np
from scipy import signal as sps


def bandpass(x: np.ndarray, fs: float, lo: float = 0.5, hi: float = 40.0, order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis."""
    hi = min(hi, 0.45 * fs)
    b, a = sps.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return sps.filtfilt(b, a, x, axis=-1).astype(np.float32)


def detect_rpeaks(x: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """Pan-Tompkins-style R-peak detector for a single lead (works on noisy wearable data).

    Band-pass 5-15 Hz -> derivative -> square -> 150 ms moving integration -> peak picking with a
    300 ms refractory period, then refinement to the largest absolute deflection within +-60 ms.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < int(fs):
        return np.zeros(0, dtype=int)
    b, a = sps.butter(2, [5 / (fs / 2), min(15, 0.45 * fs) / (fs / 2)], btype="band")
    y = sps.filtfilt(b, a, x)
    d = np.gradient(y) ** 2
    w = max(1, int(0.15 * fs))
    e = np.convolve(d, np.ones(w) / w, mode="same")
    thr = 0.25 * np.percentile(e, 99)
    pk, _ = sps.find_peaks(e, height=thr, distance=int(0.3 * fs))
    r = int(0.06 * fs)
    xc = x - np.median(x)
    pol = 1.0 if np.percentile(xc, 99.5) >= -np.percentile(xc, 0.5) else -1.0
    out = []
    for p in pk:
        lo, hi = max(0, p - r), min(len(x), p + r + 1)
        out.append(lo + int(np.argmax(pol * xc[lo:hi])))
    out = np.unique(np.array(out, dtype=int))
    # drop near-duplicates created by refinement
    if len(out) > 1:
        keep = np.concatenate([[True], np.diff(out) > int(0.25 * fs)])
        out = out[keep]
    return out


def heart_rate(x: np.ndarray, fs: float = 100.0) -> tuple[float, float]:
    """Return (mean HR in bpm, RR standard deviation in seconds); NaN when <3 beats found."""
    rp = detect_rpeaks(x, fs)
    if len(rp) < 3:
        return float("nan"), float("nan")
    rr = np.diff(rp) / fs
    rr = rr[(rr > 0.27) & (rr < 2.0)]
    if len(rr) < 2:
        return float("nan"), float("nan")
    return float(60.0 / rr.mean()), float(rr.std())


def median_beat(x: np.ndarray, fs: float = 100.0, pre: float = 0.30, post: float = 0.50):
    """Median heartbeat aligned on R. Returns (beat[pre+post samples], median RR seconds) or (None, nan)."""
    rp = detect_rpeaks(x, fs)
    a, b = int(round(pre * fs)), int(round(post * fs))
    beats = [x[p - a : p + b] for p in rp if p - a >= 0 and p + b <= len(x)]
    if len(beats) < 3:
        return None, float("nan")
    rr = np.median(np.diff(rp)) / fs
    return np.median(np.stack(beats), axis=0).astype(np.float32), float(rr)


def welch_logpsd(x: np.ndarray, fs: float = 100.0, nperseg: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Log10 power spectral density (Welch) along the last axis."""
    f, p = sps.welch(x, fs=fs, nperseg=min(nperseg, x.shape[-1]), axis=-1)
    return f, np.log10(p + 1e-12)


def snr_db(clean: np.ndarray, estimate: np.ndarray) -> float:
    """Output SNR of an estimate w.r.t. a clean reference, in dB."""
    clean = np.asarray(clean, dtype=np.float64)
    err = np.asarray(estimate, dtype=np.float64) - clean
    return float(10 * np.log10(np.sum(clean**2) / (np.sum(err**2) + 1e-12)))
