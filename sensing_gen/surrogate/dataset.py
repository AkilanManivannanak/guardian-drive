"""Simulated training data for the surrogate: impedance spectra from the finite-element solver."""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..multiphysics.electrode_skin_fem import fem_spectrum
from .space import FREQS, sample, to_configs


def simulate_one(theta) -> np.ndarray:
    tm, ec, lift = to_configs(theta)
    return fem_spectrum(FREQS, tm, ec, lift=lift)


def simulate(thetas: np.ndarray, workers: int | None = None) -> np.ndarray:
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1:
        return np.array([simulate_one(t) for t in thetas])
    with ProcessPoolExecutor(workers) as ex:
        return np.array(list(ex.map(simulate_one, thetas, chunksize=8)))


def targets(Z: np.ndarray) -> np.ndarray:
    """Complex spectra -> real targets [log10|Z| (21), phase in rad (21)]."""
    return np.concatenate([np.log10(np.abs(Z)), np.angle(Z)], axis=-1)


def from_targets(y: np.ndarray) -> np.ndarray:
    n = y.shape[-1] // 2
    return 10 ** y[..., :n] * np.exp(1j * y[..., n:])


def build(path, n: int, seed: int, expand: float = 0.0, workers: int | None = None) -> dict:
    """Generate (or load the cached) dataset of n spectra."""
    from pathlib import Path
    path = Path(path)
    if path.exists():
        d = np.load(path)
        if len(d["theta"]) == n:
            return {k: d[k] for k in d.files}
    th = sample(n, seed=seed, expand=expand)
    t0 = time.time()
    Z = simulate(th, workers)
    sec = (time.time() - t0)
    d = {"theta": th, "Z": Z, "sim_seconds_total": np.array(sec), "workers": np.array(workers or 0)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **d)
    return d


def _sim10(theta) -> complex:
    tm, ec, lift = to_configs(theta)
    return complex(fem_spectrum([10.0], tm, ec, lift=lift)[0])


def simulate_10hz(thetas: np.ndarray, workers: int | None = None) -> np.ndarray:
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1:
        return np.array([_sim10(t) for t in thetas])
    with ProcessPoolExecutor(workers) as ex:
        return np.array(list(ex.map(_sim10, thetas, chunksize=8)))
