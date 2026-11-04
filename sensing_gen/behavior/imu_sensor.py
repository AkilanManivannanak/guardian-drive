"""Physics IMU sensor model for a body-worn phone (numpy; rigid-body exact where it matters).

* Device orientation: a random rigid rotation R applied to BOTH the accelerometer (incl. gravity) and
  the gyroscope vectors, which is exactly what re-orienting a rigidly mounted phone does. With
  probability p_full the orientation is uniform on SO(3) (phone put away any way); otherwise a small
  tilt (<= small_deg).
* Calibration: per-axis accelerometer scale (1 +- 3 %) and bias, gyroscope bias (MEMS turn-on biases).
* Loose mounting: a damped 5-12 Hz oscillation of the phone in its pocket/holster, added as linear
  acceleration and the matching angular rate.
* Sampling: clock mismatch between devices (time-warp of +-5 %) and white sensor noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IMUConfig:
    fs: float = 50.0
    p_full: float = 0.5
    small_deg: float = 30.0
    acc_scale: float = 0.03
    acc_bias_g: float = 0.02
    gyro_bias: float = 0.02          # rad/s
    p_wobble: float = 0.5
    wobble_g: float = 0.05
    p_warp: float = 0.5
    warp: float = 0.05
    acc_noise_g: float = 0.01
    gyro_noise: float = 0.01


def random_rotation(rng: np.random.Generator, n: int, max_deg: float | None = None) -> np.ndarray:
    """(n, 3, 3) rotations: uniform on SO(3) if max_deg is None, else axis uniform, angle U(0, max_deg)."""
    if max_deg is None:
        q = rng.standard_normal((n, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        w, x, y, z = q.T
    else:
        ax = rng.standard_normal((n, 3))
        ax /= np.linalg.norm(ax, axis=1, keepdims=True)
        ang = np.deg2rad(rng.uniform(0, max_deg, n))
        w = np.cos(ang / 2)
        x, y, z = (ax * np.sin(ang / 2)[:, None]).T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def rotate(x: np.ndarray, R: np.ndarray) -> np.ndarray:
    """x (N, 6, T) = [acc xyz, gyro xyz]; apply R (N, 3, 3) to both triads."""
    acc = np.einsum("nij,njt->nit", R, x[:, :3])
    gyr = np.einsum("nij,njt->nit", R, x[:, 3:])
    return np.concatenate([acc, gyr], 1).astype(np.float32)


class IMUSensorModel:
    def __init__(self, cfg: IMUConfig | None = None):
        self.cfg = cfg or IMUConfig()

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        c = self.cfg
        x = np.asarray(x, np.float32).copy()
        n, _, T = x.shape
        full = rng.random(n) < c.p_full
        R = np.where(full[:, None, None], random_rotation(rng, n), random_rotation(rng, n, c.small_deg))
        # time-warp (clock mismatch), linear interpolation along time
        warp = np.where(rng.random(n) < c.p_warp, rng.uniform(1 - c.warp, 1 + c.warp, n), 1.0)
        t = np.arange(T)
        for i in np.nonzero(warp != 1.0)[0]:
            src = np.clip((t - T / 2) * warp[i] + T / 2, 0, T - 1)
            x[i] = np.stack([np.interp(src, t, ch) for ch in x[i]])
        # loose mounting: damped oscillation in the device frame (acc) + matching angular rate
        wob = rng.random(n) < c.p_wobble
        if wob.any():
            tt = t / c.fs
            f = rng.uniform(5, 12, n)[:, None]
            zeta = rng.uniform(0.05, 0.3, n)[:, None]
            t0 = rng.uniform(0, tt[-1], n)[:, None]
            env = np.exp(-zeta * 2 * np.pi * f * np.clip(tt[None] - t0, 0, None)) * (tt[None] >= t0)
            osc = env * np.sin(2 * np.pi * f * (tt[None] - t0))
            d = rng.standard_normal((n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
            amp = (c.wobble_g * wob)[:, None, None]
            x[:, :3] += amp * d[:, :, None] * osc[:, None]
            x[:, 3:] += 2 * np.pi * f[:, :, None] * amp * 0.1 * d[:, :, None] * np.cos(2 * np.pi * f * (tt[None] - t0))[:, None] * env[:, None]
        x = rotate(x, R)
        x[:, :3] = x[:, :3] * (1 + c.acc_scale * rng.standard_normal((n, 3, 1))) + c.acc_bias_g * rng.standard_normal((n, 3, 1))
        x[:, 3:] = x[:, 3:] + c.gyro_bias * rng.standard_normal((n, 3, 1))
        x[:, :3] += c.acc_noise_g * rng.standard_normal((n, 3, T))
        x[:, 3:] += c.gyro_noise * rng.standard_normal((n, 3, T))
        return x.astype(np.float32)
