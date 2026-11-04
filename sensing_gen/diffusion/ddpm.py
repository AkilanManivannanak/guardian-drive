"""Gaussian diffusion (Ho et al. 2020) with cosine schedule, DDIM sampling (Song et al. 2021)
and classifier-free guidance (Ho & Salimans 2022)."""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from .physics_guidance import limit_cycle_residual


def cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0).float()


class GaussianDiffusion(nn.Module):
    def __init__(self, model: nn.Module, T: int = 1000, p_uncond: float = 0.1):
        super().__init__()
        self.model, self.T, self.p_uncond = model, T, p_uncond
        self.register_buffer("ab", cosine_alpha_bar(T))

    def q_sample(self, x0, t, noise):
        ab = self.ab[t][:, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def loss(self, x0: torch.Tensor, y: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """x0 (B, 1, L); y (B,) class ids; hr (B,) bpm. eps-prediction MSE with label dropout."""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        drop = torch.rand(B, device=x0.device) < self.p_uncond
        y = torch.where(drop, torch.full_like(y, self.model.n_classes), y)
        eps = self.model(self.q_sample(x0, t, noise), t, y, hr)
        return ((eps - noise) ** 2).mean()

    @torch.no_grad()
    def eps_cfg(self, x, t, y, hr, w: float):
        if w == 0:
            return self.model(x, t, y, hr)
        null = torch.full_like(y, self.model.n_classes)
        e = self.model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null]), torch.cat([hr, hr]))
        ec, eu = e.chunk(2)
        return eu + (1 + w) * (ec - eu)

    @torch.no_grad()
    def ddim_sample(self, n: int, length: int, y: torch.Tensor, hr: torch.Tensor, steps: int = 50,
                    w: float = 2.0, eta: float = 0.0, gen: torch.Generator | None = None,
                    clip: float | None = 12.0, spacing: str = "trailing",
                    threshold: str = "static", phys: float = 0.0, fs: float = 100.0) -> torch.Tensor:
        """DDIM sampler.

        phys > 0 applies physics-informed limit-cycle guidance (see physics_guidance.py) to the x0
        estimate at every step: x0 <- x0 - phys * (x0 - P_tau x0), tau = one RR interval at the
        conditioned heart rate. phys = 0 reproduces the unguided sampler exactly.

        spacing="trailing" starts at t = T-1, where the cosine schedule has alpha_bar ~ 1e-9, so the
        x0 estimate amplifies epsilon errors by ~2e4. "leading" (t = T - T/steps, ..., 0) starts
        where alpha_bar ~ 9e-4 (amplification ~34x). threshold="dynamic" clamps x0 per sample at its 99.5th percentile
        magnitude (bounded to [4, clip]) instead of a fixed +-clip (Saharia et al., 2022).
        """
        dev = self.ab.device
        x = torch.randn(n, getattr(self.model, "c_in", 1), length, generator=gen).to(dev)
        if spacing == "leading":
            ts = torch.arange(0, self.T, self.T // steps, device=dev).flip(0).long()
        else:
            ts = torch.linspace(self.T - 1, 0, steps, device=dev).long()
        for i, t in enumerate(ts):
            tt = torch.full((n,), int(t), device=dev, dtype=torch.long)
            eps = self.eps_cfg(x, tt, y, hr, w)
            ab = self.ab[t]
            ab_prev = self.ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
            x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            if phys > 0:
                x0 = x0 - phys * limit_cycle_residual(x0, hr, fs)
                if clip is None:
                    eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            if clip is not None:
                if threshold == "dynamic":
                    s = torch.quantile(x0.abs().flatten(1).float().cpu(), 0.995, dim=1).clamp(4.0, clip).to(dev)
                    s = s[:, None, None]
                    x0 = torch.maximum(torch.minimum(x0, s), -s)
                else:
                    x0 = x0.clamp(-clip, clip)
                eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            sigma = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).clamp(min=0).sqrt()
            dir_xt = (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps
            x = ab_prev.sqrt() * x0 + dir_xt
            if eta > 0 and i + 1 < len(ts):
                x = x + sigma * torch.randn(x.shape, generator=gen).to(dev)
        return x


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for s, b in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(b)
