"""Calibrate the ECG dynamical model to real data by gradient descent *through the simulator*.

For every real record we extract the R-aligned median beat and fit the 16 physical parameters
(5 event angles, amplitudes, widths + DC offset) of the McSharry model in parallel. The fitted
parameter vectors form an empirical, class-conditional "physics prior" (ParamBank) that the
simulator samples from, which narrows the sim-to-real gap compared with hand-set defaults.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data.signal_utils import bandpass, median_beat
from .ecg_ode import (DEFAULT_A, DEFAULT_B, DEFAULT_THETA, ECGParams, ECGSimulator,
                      r_wave_gain)


@dataclass
class ParamBank:
    theta: np.ndarray   # (N, 5)
    a: np.ndarray       # (N, 5)
    b: np.ndarray       # (N, 5)
    label: np.ndarray   # (N,)
    hr: np.ndarray      # (N,)
    rr_std: np.ndarray  # (N,)
    r2: np.ndarray      # (N,) fit quality of the median beat

    def save(self, path) -> None:
        np.savez(path, **self.__dict__)

    @staticmethod
    def load(path) -> "ParamBank":
        z = np.load(path)
        return ParamBank(**{k: z[k] for k in z.files})


@torch.no_grad()
def solve_amplitudes(sim: ECGSimulator, theta: torch.Tensor, b: torch.Tensor, rr: torch.Tensor,
                     x: torch.Tensor, a_scale: torch.Tensor, ridge: float = 1e-4):
    """Variable projection: the beat is *linear* in the amplitudes a_i and the offset, so for fixed
    (theta, b) solve them exactly by ridge least squares. Returns (a_n (N, 5), offset (N,))."""
    N, K = theta.shape
    eye = torch.eye(K, device=theta.device)
    a = (eye[None] * a_scale[:, :, None]).reshape(N * K, K)          # one active wave per row
    p = ECGParams(theta.repeat_interleave(K, 0), a, b.repeat_interleave(K, 0),
                  torch.zeros(N * K, device=theta.device))
    Z = sim.simulate_beat(p, rr.repeat_interleave(K, 0)).reshape(N, K, -1)   # (N, K, L)
    A = torch.cat([Z, torch.ones_like(Z[:, :1])], 1).transpose(1, 2)        # (N, L, K+1)
    AtA = A.transpose(1, 2) @ A
    reg = ridge * AtA.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * torch.diag(
        torch.tensor([1.0] * K + [0.0], device=theta.device))[None]
    sol = torch.linalg.solve(AtA + reg + 1e-9 * torch.eye(K + 1, device=theta.device)[None],
                             A.transpose(1, 2) @ x[:, :, None])[..., 0]
    return sol[:, :K], sol[:, K]


def fit_beats(beats: np.ndarray, rr: np.ndarray, steps: int = 800, lr: float = 0.03,
              prior: float = 1e-3, device: str = "cpu", verbose: bool = False):
    """Fit model parameters to N median beats in parallel. Returns (ECGParams, r2 (N,), fitted (N, L))."""
    sim = ECGSimulator()
    x = torch.tensor(beats, dtype=torch.float32, device=device)
    rr_t = torch.tensor(rr, dtype=torch.float32, device=device)
    N = x.shape[0]
    th0 = torch.tensor(DEFAULT_THETA, device=device).repeat(N, 1)
    lb0 = torch.log(torch.tensor(DEFAULT_B, device=device)).repeat(N, 1)
    # amplitude initialisation: default shape scaled to each beat's R amplitude
    g = r_wave_gain()
    amp = (x.max(1).values - x.median(1).values).clamp(min=0.05)
    # amplitudes are optimised in units of the beat's own scale, so Adam's step size is
    # meaningful for every beat (and a wave can flip sign, e.g. an inverted T wave)
    a_scale = (g * amp)[:, None]

    dth = torch.zeros(N, 5, device=device, requires_grad=True)
    lb = lb0.clone().requires_grad_(True)
    # exact least-squares amplitudes for the default geometry: avoids the local minimum where a
    # wrongly-signed wave (e.g. an inverted T) is "fixed" by shrinking its width instead of flipping
    a_init, off_init = solve_amplitudes(sim, th0, torch.exp(lb0), rr_t, x, a_scale)
    a_n = a_init.clone().requires_grad_(True)
    off = off_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([dth, a_n, lb, off], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for it in range(steps):
        if it > 0 and it % 100 == 0:  # periodic re-projection of the linear parameters
            a_p, o_p = solve_amplitudes(sim, (th0 + dth).detach(), torch.exp(lb).clamp(0.02, 1.2).detach(),
                                        rr_t, x, a_scale)
            with torch.no_grad():
                a_n.copy_(a_p); off.copy_(o_p)
        a = a_n * a_scale
        p = ECGParams(th0 + dth, a, torch.exp(lb).clamp(0.02, 1.2), off)
        y = sim.simulate_beat(p, rr_t)
        mse = ((y - x) ** 2).mean(1)
        reg = prior * ((dth**2).sum(1) + ((lb - lb0) ** 2).sum(1))
        loss = (mse + reg).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and it % 200 == 0:
            print(f"  calib it {it:4d}  mse {mse.mean().item():.4f}")
    with torch.no_grad():
        p = ECGParams(th0 + dth, a_n * a_scale, torch.exp(lb).clamp(0.02, 1.2), off).detach()
        y = sim.simulate_beat(p, rr_t)
        sse = ((y - x) ** 2).sum(1)
        sst = ((x - x.mean(1, keepdim=True)) ** 2).sum(1) + 1e-8
        r2 = (1 - sse / sst).cpu().numpy()
    return p, r2, y.cpu().numpy()


def build_bank(x: np.ndarray, y: np.ndarray, hr: np.ndarray, rr_std: np.ndarray,
               fs: float = 100.0, steps: int = 800, device: str = "cpu", verbose=False) -> ParamBank:
    beats, rrs, keep = [], [], []
    for i in range(len(x)):
        b, rr = median_beat(x[i], fs)
        if b is not None and np.isfinite(hr[i]) and 0.3 < rr < 2.0:
            beats.append(b)
            rrs.append(rr)
            keep.append(i)
    keep = np.array(keep)
    beats_a, rr_a = np.stack(beats), np.array(rrs)
    th, aa, bb, r2s = [], [], [], []
    for s in range(0, len(beats_a), 512):  # chunk to bound memory
        p, r2, _ = fit_beats(beats_a[s : s + 512], rr_a[s : s + 512], steps=steps, device=device,
                             verbose=verbose)
        th.append(p.theta.cpu().numpy()); aa.append(p.a.cpu().numpy())
        bb.append(p.b.cpu().numpy()); r2s.append(r2)
        if verbose:
            print(f"  calibrated {min(s + 512, len(beats_a))}/{len(beats_a)} beats")
    rs = np.nan_to_num(rr_std[keep], nan=0.03)
    return ParamBank(np.concatenate(th), np.concatenate(aa), np.concatenate(bb), y[keep],
                     hr[keep], rs, np.concatenate(r2s))


def sample_from_bank(bank: ParamBank, labels: np.ndarray, rng: np.random.Generator,
                     min_r2: float = 0.5, duration_s: float = 10.0, fs: float = 100.0,
                     seed: int = 0) -> np.ndarray:
    """Simulate one 10-s record per requested label by resampling + jittering calibrated parameters.

    The simulated signal is passed through the same 0.5-40 Hz acquisition filter as the real data.
    """
    sim = ECGSimulator(fs_out=fs)
    good = bank.r2 >= min_r2
    out = np.zeros((len(labels), int(duration_s * fs)), dtype=np.float32)
    gen = torch.Generator().manual_seed(seed)
    for c in np.unique(labels):
        idx_pool = np.nonzero(good & (bank.label == c))[0]
        if len(idx_pool) == 0:
            idx_pool = np.nonzero(bank.label == c)[0]
        rows = np.nonzero(labels == c)[0]
        pick = rng.choice(idx_pool, size=len(rows))
        n = len(rows)
        theta = bank.theta[pick] + rng.normal(0, 0.02, (n, 5))
        a = bank.a[pick] * (1 + rng.normal(0, 0.05, (n, 5)))
        b = bank.b[pick] * np.exp(rng.normal(0, 0.05, (n, 5)))
        p = ECGParams(torch.tensor(theta, dtype=torch.float32), torch.tensor(a, dtype=torch.float32),
                      torch.tensor(b, dtype=torch.float32), torch.zeros(n))
        hr = torch.tensor(bank.hr[pick] * (1 + rng.normal(0, 0.03, n)), dtype=torch.float32).clamp(35, 180)
        rs = torch.tensor(np.clip(bank.rr_std[pick], 0.005, 0.15), dtype=torch.float32)
        resp = torch.tensor(rng.uniform(0.15, 0.35, n), dtype=torch.float32)
        am = torch.tensor(rng.uniform(0.0, 0.1, n), dtype=torch.float32)
        bw = torch.tensor(rng.uniform(0.0, 0.3, n), dtype=torch.float32)
        for s in range(0, n, 256):
            sl = slice(s, s + 256)
            ps = ECGParams(p.theta[sl], p.a[sl], p.b[sl], p.offset[sl])
            with torch.no_grad():
                z = sim.simulate(ps, duration_s, hr[sl], rs[sl], resp[sl], bw_amp=bw[sl],
                                 am_depth=am[sl], gen=gen)
            out[rows[sl]] = z.numpy()
    return bandpass(out, fs)
