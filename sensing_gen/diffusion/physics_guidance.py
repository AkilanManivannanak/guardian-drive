"""Physics-informed guidance for the ECG diffusion sampler.

The McSharry ECG model is a limit-cycle oscillator: its trajectory travels around the unit circle once
per RR interval and the ECG is the z-coordinate, a function of the cycle phase. A sample conditioned on
heart rate HR should therefore be (approximately) invariant to a shift by one period tau = 60 fs / HR.

The guidance energy is the limit-cycle residual

    E(x) = 1/4 * sum_t (x(t) - x(t + tau))^2 ,

whose gradient is dE/dx(t) = x(t) - 0.5 (x(t + tau) + x(t - tau)) = x - P_tau x  (interior points), so a
gradient step of size lam on the sampler's x0-estimate is x0 <- x0 - lam (x0 - P_tau x0). lam = 1 projects
fully onto locally periodic signals; small lam keeps beat-to-beat variability. Applied in x0-space, it
needs no backpropagation through the denoiser and adds negligible cost.
"""
from __future__ import annotations

import torch


def shift_frac(x: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x (B, C, L) sampled at x(t + s_b) with linear interpolation; s (B,) in samples, may be fractional.
    Returns the shifted signal and a {0, 1} mask of positions where t + s lies inside [0, L - 1]."""
    B, C, L = x.shape
    t = torch.arange(L, device=x.device, dtype=x.dtype)[None, :] + s[:, None].to(x.dtype)   # (B, L)
    valid = ((t >= 0) & (t <= L - 1)).to(x.dtype)
    tc = t.clamp(0, L - 1)
    i0 = tc.floor().long().clamp(max=L - 2)
    fr = (tc - i0.to(x.dtype))[:, None, :]
    i0 = i0[:, None, :].expand(B, C, L)
    out = torch.gather(x, 2, i0) * (1 - fr) + torch.gather(x, 2, i0 + 1) * fr
    return out, valid[:, None, :]


def limit_cycle_residual(x: torch.Tensor, hr_bpm: torch.Tensor, fs: float = 100.0) -> torch.Tensor:
    """r = x - P_tau x, with P_tau the average of the available one-period neighbours (edges use one)."""
    tau = fs * 60.0 / hr_bpm.clamp(30.0, 200.0)
    xp, mp = shift_frac(x, tau)
    xm, mm = shift_frac(x, -tau)
    cnt = (mp + mm).clamp(min=1.0)
    proj = (xp * mp + xm * mm) / cnt
    proj = torch.where((mp + mm) > 0, proj, x)
    return x - proj


def limit_cycle_energy(x: torch.Tensor, hr_bpm: torch.Tensor, fs: float = 100.0) -> torch.Tensor:
    """Per-sample normalised residual energy ||x - P_tau x||^2 / ||x - mean||^2 (0 = perfectly periodic)."""
    r = limit_cycle_residual(x, hr_bpm, fs)
    xc = x - x.mean(-1, keepdim=True)
    return (r ** 2).flatten(1).sum(1) / (xc ** 2).flatten(1).sum(1).clamp(min=1e-8)
