"""Training / sampling helpers for the conditional ECG diffusion model."""
from __future__ import annotations

import time

import numpy as np
import torch

from .ddpm import EMA, GaussianDiffusion
from .unet1d import UNet1D


def make_model(base: int = 32) -> GaussianDiffusion:
    return GaussianDiffusion(UNet1D(base=base, mults=(1, 2, 4), n_classes=2, hr_cond=True))


def _clean_hr(hr: np.ndarray) -> np.ndarray:
    hr = np.asarray(hr, dtype=np.float32).copy()
    med = np.nanmedian(hr) if np.isfinite(hr).any() else 70.0
    hr[~np.isfinite(hr)] = med
    return hr


def train_diffusion(x: np.ndarray, y: np.ndarray, hr: np.ndarray, steps: int, device: str,
                    batch: int = 64, lr: float = 2e-4, init: dict | None = None, seed: int = 0,
                    log_every: int = 500, base: int = 32, tag: str = "") -> tuple[GaussianDiffusion, list]:
    """Train (or fine-tune from `init` state_dict) and return the EMA model + loss log."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    diff = make_model(base).to(device)
    if init is not None:
        diff.load_state_dict(init)
    ema = EMA(diff, decay=0.999)
    opt = torch.optim.AdamW(diff.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500))
    X = torch.tensor(x, dtype=torch.float32)[:, None]
    Y = torch.tensor(y, dtype=torch.long)
    H = torch.tensor(_clean_hr(hr), dtype=torch.float32)
    log, t0, run = [], time.time(), 0.0
    for step in range(1, steps + 1):
        idx = torch.from_numpy(rng.integers(0, len(X), batch))
        xb = X[idx]
        loss = diff.loss(xb.to(device), Y[idx].to(device), H[idx].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diff.parameters(), 1.0)
        opt.step()
        sched.step()
        ema.update(diff)
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % log_every == 0 or step == steps or step == 100:
            el = time.time() - t0
            print(f"[diffusion{tag}] step {step}/{steps} loss {run:.4f}  {step / el:.1f} it/s", flush=True)
            log.append({"step": step, "loss": run})
    return ema.shadow, log


@torch.no_grad()
def generate(diff: GaussianDiffusion, labels: np.ndarray, hrs: np.ndarray, device: str,
             length: int = 1000, steps: int = 50, w: float = 2.0, batch: int = 256,
             seed: int = 0, spacing: str = "trailing", threshold: str = "static",
             phys: float = 0.0) -> np.ndarray:
    diff.eval()
    gen = torch.Generator().manual_seed(seed)
    out = []
    hrs = _clean_hr(hrs)
    for s in range(0, len(labels), batch):
        yb = torch.tensor(labels[s : s + batch], dtype=torch.long, device=device)
        hb = torch.tensor(hrs[s : s + batch], dtype=torch.float32, device=device)
        xs = diff.ddim_sample(len(yb), length, yb, hb, steps=steps, w=w, gen=gen, spacing=spacing,
                              threshold=threshold, phys=phys)
        out.append(xs[:, 0].float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)
