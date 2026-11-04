"""Deep-ensemble MLP surrogate: electrode/tissue parameters -> impedance spectrum, with uncertainty.

Each member predicts a mean and a log-variance for every output (Lakshminarayanan et al., 2017) and is
trained with the beta-NLL loss (Seitzer et al., 2022) after an MSE warm-up. The ensemble prediction is
the Gaussian mixture: mean = average of member means, variance = average (sigma_k^2 + mu_k^2) - mean^2,
so it captures both noise-free model error (aleatoric heads, ~0 here) and epistemic disagreement.

hybrid=True makes it a physics-informed residual model: the network learns only the correction to the
analytic contact-impedance prior (space.contact_prior), i.e. the tissue spreading impedance and the
geometry/lift-off interaction, which is smoother and much more data-efficient than the full spectrum.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .space import contact_prior, to_unit, unit_to_phys_torch


class Member(nn.Module):
    def __init__(self, d_in: int, d_out: int, width: int = 256, depth: int = 3):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        self.body = nn.Sequential(*layers)
        self.mu = nn.Linear(d, d_out)
        self.logvar = nn.Linear(d, d_out)
        nn.init.constant_(self.logvar.bias, -4.0)

    def forward(self, u):
        h = self.body(u)
        return self.mu(h), self.logvar(h).clamp(-14.0, 4.0)


class Surrogate(nn.Module):
    """Wraps an ensemble with the input/output normalisation, so it can be saved and reused."""

    def __init__(self, d_in: int, d_out: int, k: int = 5, width: int = 256, depth: int = 3,
                 hybrid: bool = True):
        super().__init__()
        self.hybrid = hybrid
        self.members = nn.ModuleList([Member(d_in, d_out, width, depth) for _ in range(k)])
        self.register_buffer("y_mean", torch.zeros(d_out))
        self.register_buffer("y_std", torch.ones(d_out))
        self.cfg = {"d_in": d_in, "d_out": d_out, "k": k, "width": width, "depth": depth, "hybrid": hybrid}

    def prior(self, u: torch.Tensor) -> torch.Tensor:
        if not self.hybrid:
            return torch.zeros(u.shape[0], self.y_mean.shape[0], dtype=u.dtype, device=u.device)
        return contact_prior(unit_to_phys_torch(u), torch)

    def forward(self, u: torch.Tensor):
        """u in [0, 1]^d (see space.to_unit). Returns mixture mean and std in *physical target units*
        (log10|Z| and phase in rad), plus the per-member means."""
        x = 2 * u - 1
        mus, vs = zip(*[m(x) for m in self.members])
        mu = torch.stack(mus)            # (K, B, D) standardised
        var = torch.stack(vs).exp()
        m = mu.mean(0)
        v = (var + mu ** 2).mean(0) - m ** 2
        pr = self.prior(u)
        return (m * self.y_std + self.y_mean + pr, v.clamp(min=1e-12).sqrt() * self.y_std,
                mu * self.y_std + self.y_mean + pr)

    @torch.no_grad()
    def predict(self, theta: np.ndarray, device: str = "cpu", batch: int = 8192):
        u = torch.tensor(to_unit(theta), dtype=torch.float32)
        out_m, out_s = [], []
        for s in range(0, len(u), batch):
            m, sd, _ = self(u[s:s + batch].to(device))
            out_m.append(m.cpu().numpy()); out_s.append(sd.cpu().numpy())
        return np.concatenate(out_m), np.concatenate(out_s)


def train_surrogate(theta: np.ndarray, y: np.ndarray, k: int = 5, steps: int = 6000, batch: int = 256,
                    lr: float = 2e-3, beta: float = 0.5, warmup: float = 0.3, width: int = 256,
                    depth: int = 3, device: str = "cpu", seed: int = 0, log_every: int = 0,
                    hybrid: bool = True) -> Surrogate:
    torch.manual_seed(seed)
    u = torch.tensor(to_unit(theta), dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    model = Surrogate(u.shape[1], yt.shape[1], k, width, depth, hybrid).to(device)
    if hybrid:
        yt = yt - torch.tensor(contact_prior(theta), dtype=torch.float32, device=device)
    model.y_mean.copy_(yt.mean(0)); model.y_std.copy_(yt.std(0).clamp(min=1e-6))
    ys = (yt - model.y_mean) / model.y_std
    x = 2 * u - 1
    n = len(x)
    for j, mem in enumerate(model.members):
        g = torch.Generator(device="cpu").manual_seed(seed * 100 + j)
        opt = torch.optim.AdamW(mem.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
        for it in range(steps):
            idx = torch.randint(0, n, (min(batch, n),), generator=g).to(device)
            mu, lv = mem(x[idx])
            err2 = (mu - ys[idx]) ** 2
            if it < warmup * steps:
                loss = err2.mean()
            else:
                var = lv.exp()
                nll = 0.5 * (lv + err2 / var)
                loss = (nll * var.detach() ** beta).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            if log_every and it % log_every == 0:
                print(f"[surrogate] member {j} step {it} loss {loss.item():.5f}", flush=True)
    return model.eval()


def load_surrogate(path, device: str = "cpu") -> Surrogate:
    ck = torch.load(path, map_location=device)
    m = Surrogate(**ck["cfg"]).to(device)
    m.load_state_dict(ck["state"])
    return m.eval()


def save_surrogate(model: Surrogate, path) -> None:
    torch.save({"cfg": model.cfg, "state": model.state_dict()}, path)
