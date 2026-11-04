"""Multi-channel activity classifier and conditional IMU diffusion model (PyTorch)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..diffusion.ddpm import EMA, GaussianDiffusion
from ..diffusion.unet1d import UNet1D
from ..eval.models import ResNet1D


def macro_f1(y: np.ndarray, p: np.ndarray, k: int) -> float:
    f = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


@torch.no_grad()
def predict_logits(model, x: np.ndarray, device: str, batch: int = 512) -> np.ndarray:
    model.eval()
    return np.concatenate([model(torch.tensor(x[s: s + batch], device=device)).cpu().numpy()
                           for s in range(0, len(x), batch)])


@torch.no_grad()
def embed(model, x: np.ndarray, device: str, batch: int = 512) -> np.ndarray:
    model.eval()
    return np.concatenate([model.features(torch.tensor(x[s: s + batch], device=device)).cpu().numpy()
                           for s in range(0, len(x), batch)])


def train_har(x, y, xv, yv, device, steps=2000, seed=0, augment=None, p_aug=0.8, extra=None, p_extra=0.5,
              batch=128, lr=2e-3, eval_every=250, k=6):
    """Train a 1-D ResNet on (N, 6, T) windows; select the checkpoint on validation macro-F1.
    augment(x_numpy, rng) -> x_numpy is applied to a random p_aug fraction of each batch;
    extra = (x_syn, y_syn) replaces a p_extra fraction of each batch with synthetic windows."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = ResNet1D(n_out=k, c_in=x.shape[1]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    best, state = -1.0, None
    lossf = nn.CrossEntropyLoss()
    for it in range(1, steps + 1):
        m.train()
        idx = rng.integers(0, len(x), batch)
        xb, yb = x[idx].copy(), y[idx].copy()
        if extra is not None:
            sel = rng.random(batch) < p_extra
            if sel.any():
                j = rng.integers(0, len(extra[0]), int(sel.sum()))
                xb[sel], yb[sel] = extra[0][j], extra[1][j]
        if augment is not None:
            sel = rng.random(batch) < p_aug
            if sel.any():
                xb[sel] = augment(xb[sel], rng)
        out = m(torch.tensor(xb, device=device))
        loss = lossf(out, torch.tensor(yb, device=device))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % eval_every == 0 or it == steps:
            f1 = macro_f1(yv, predict_logits(m, xv, device).argmax(1), k)
            if f1 > best:
                best, state = f1, {kk: v.detach().clone() for kk, v in m.state_dict().items()}
    m.load_state_dict(state)
    return m.eval(), best


class IMUDiffusion:
    """Class-conditional DDPM over per-channel standardised (6, 128) IMU windows."""

    def __init__(self, n_classes=6, c_in=6, base=32, device="cpu"):
        self.diff = GaussianDiffusion(UNet1D(base=base, mults=(1, 2, 4), n_classes=n_classes, hr_cond=False,
                                             c_in=c_in)).to(device)
        self.device, self.mu, self.sd = device, None, None

    def fit(self, x, y, steps=6000, batch=64, lr=2e-4, seed=0, log_every=1000, tag=""):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.mu = x.mean((0, 2), keepdims=True); self.sd = x.std((0, 2), keepdims=True) + 1e-6
        X = torch.tensor((x - self.mu) / self.sd, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.long)
        ema = EMA(self.diff, decay=0.999)
        opt = torch.optim.AdamW(self.diff.parameters(), lr=lr, weight_decay=1e-4)
        hr0 = torch.zeros(batch, device=self.device)
        run = None
        for it in range(1, steps + 1):
            idx = torch.from_numpy(rng.integers(0, len(X), batch))
            loss = self.diff.loss(X[idx].to(self.device), Y[idx].to(self.device), hr0)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.diff.parameters(), 1.0); opt.step(); ema.update(self.diff)
            run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
            if it % log_every == 0 or it == steps:
                print(f"[behavior{tag}] diffusion step {it}/{steps} loss {run:.4f}", flush=True)
        self.diff = ema.shadow
        return self

    @torch.no_grad()
    def sample(self, labels: np.ndarray, length=128, steps=50, w=1.0, seed=0, batch=512) -> np.ndarray:
        gen = torch.Generator().manual_seed(seed)
        out = []
        for s in range(0, len(labels), batch):
            yb = torch.tensor(labels[s: s + batch], dtype=torch.long, device=self.device)
            hb = torch.zeros(len(yb), device=self.device)
            xs = self.diff.ddim_sample(len(yb), length, yb, hb, steps=steps, w=w, gen=gen,
                                       spacing="leading", threshold="dynamic")
            out.append(xs.float().cpu().numpy())
        return (np.concatenate(out) * self.sd + self.mu).astype(np.float32)
