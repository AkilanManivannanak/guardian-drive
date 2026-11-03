"""1-D U-Net denoiser with FiLM conditioning on (diffusion time, class label, heart rate)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal(x: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=x.device) / half)
    args = x.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, emb: int, k: int = 7, groups: int = 8):
        super().__init__()
        self.n1 = nn.GroupNorm(groups, c_in)
        self.c1 = nn.Conv1d(c_in, c_out, k, padding=k // 2)
        self.film = nn.Linear(emb, 2 * c_out)
        self.n2 = nn.GroupNorm(groups, c_out)
        self.c2 = nn.Conv1d(c_out, c_out, k, padding=k // 2)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.film(e)[..., None].chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(F.silu(h))
        return h + self.skip(x)


class SelfAttention1d(nn.Module):
    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        h = self.norm(x).transpose(1, 2)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.transpose(1, 2)


class UNet1D(nn.Module):
    """Denoiser eps_theta(x_t, t, c). Input length must be divisible by 2**(len(mults)-1)."""

    def __init__(self, base: int = 64, mults=(1, 2, 4), n_classes: int = 2, emb: int = 256,
                 hr_cond: bool = True):
        super().__init__()
        self.n_classes = n_classes
        self.hr_cond = hr_cond
        self.t_mlp = nn.Sequential(nn.Linear(base, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.cls_emb = nn.Embedding(n_classes + 1, emb)          # last index = null (CFG)
        self.hr_mlp = nn.Sequential(nn.Linear(64, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.null_hr = nn.Parameter(torch.zeros(emb))
        self.base = base
        self.inp = nn.Conv1d(1, base, 7, padding=3)
        chans = [base * m for m in mults]
        self.down = nn.ModuleList()
        c = base
        skips = [c]
        for i, co in enumerate(chans):
            blocks = nn.ModuleList([ResBlock(c, co, emb), ResBlock(co, co, emb)])
            c = co
            skips += [c, c]
            ds = nn.Conv1d(c, c, 4, stride=2, padding=1) if i < len(chans) - 1 else nn.Identity()
            self.down.append(nn.ModuleList([blocks, ds]))
            if i < len(chans) - 1:
                skips.append(c)
        self.mid1 = ResBlock(c, c, emb)
        self.mid_attn = SelfAttention1d(c)
        self.mid2 = ResBlock(c, c, emb)
        self.up = nn.ModuleList()
        for i, co in reversed(list(enumerate(chans))):
            n_blocks = 3  # 2 res-block skips + 1 (input / downsample) skip per level
            blocks = nn.ModuleList()
            for _ in range(n_blocks):
                blocks.append(ResBlock(c + skips.pop(), co, emb))
                c = co
            us = nn.Upsample(scale_factor=2, mode="nearest") if i > 0 else nn.Identity()
            self.up.append(nn.ModuleList([blocks, us]))
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv1d(c, 1, 7, padding=3))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def embed(self, t, y, hr):
        e = self.t_mlp(sinusoidal(t, self.base)) + self.cls_emb(y)
        if self.hr_cond:
            he = self.hr_mlp(sinusoidal(hr.clamp(0, 250), 64, max_period=1000.0))
            null = (y == self.n_classes)[:, None]
            e = e + torch.where(null, self.null_hr[None].expand_as(he), he)
        return e

    def forward(self, x, t, y, hr):
        e = self.embed(t, y, hr)
        h = self.inp(x)
        hs = [h]
        for blocks, ds in self.down:
            for b in blocks:
                h = b(h, e)
                hs.append(h)
            if not isinstance(ds, nn.Identity):
                h = ds(h)
                hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, e)), e)
        for blocks, us in self.up:
            for b in blocks:
                h = b(torch.cat([h, hs.pop()], dim=1), e)
            h = us(h)
        return self.out(h)
