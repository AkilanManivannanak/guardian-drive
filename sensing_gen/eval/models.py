"""Small 1-D ResNet classifier + training loop used for downstream utility and fidelity scoring."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import auroc  # noqa: F401  (re-exported)


class Block(nn.Module):
    def __init__(self, ci: int, co: int, stride: int):
        super().__init__()
        self.c1 = nn.Conv1d(ci, co, 7, stride=stride, padding=3, bias=False)
        self.b1 = nn.BatchNorm1d(co)
        self.c2 = nn.Conv1d(co, co, 7, padding=3, bias=False)
        self.b2 = nn.BatchNorm1d(co)
        self.sk = nn.Sequential(nn.Conv1d(ci, co, 1, stride=stride, bias=False), nn.BatchNorm1d(co))

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(h)) + self.sk(x))


class ResNet1D(nn.Module):
    def __init__(self, n_out: int = 1, width: int = 32, dropout: float = 0.2):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv1d(1, w, 15, stride=2, padding=7, bias=False),
                                  nn.BatchNorm1d(w), nn.ReLU())
        self.layers = nn.Sequential(Block(w, w, 2), Block(w, 2 * w, 2), Block(2 * w, 4 * w, 2),
                                    Block(4 * w, 4 * w, 2))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * w, n_out)

    def features(self, x):
        return self.layers(self.stem(x)).mean(-1)

    def forward(self, x):
        return self.fc(self.drop(self.features(x)))


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: str, batch: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(x), batch):
        xb = torch.tensor(x[s : s + batch], dtype=torch.float32, device=device)[:, None]
        out.append(torch.sigmoid(model(xb)[:, 0]).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def embed(model: ResNet1D, x: np.ndarray, device: str, batch: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(x), batch):
        xb = torch.tensor(x[s : s + batch], dtype=torch.float32, device=device)[:, None]
        out.append(model.features(xb).cpu().numpy())
    return np.concatenate(out)


def train_classifier(x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                     device: str, steps: int = 2000, batch: int = 128, lr: float = 1e-3,
                     extra: Optional[tuple[np.ndarray, np.ndarray]] = None, p_extra: float = 0.5,
                     augment: Optional[Callable[[torch.Tensor, torch.Generator], torch.Tensor]] = None,
                     p_aug: float = 0.8, eval_every: int = 250, seed: int = 0,
                     x_val_aug: Optional[np.ndarray] = None) -> tuple[ResNet1D, dict]:
    """Train a binary classifier; model selection on validation AUROC (clean, or `x_val_aug` if given,
    averaged with clean). `extra` = synthetic (x, y) mixed into each batch with probability p_extra."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    model = ResNet1D().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    pos = max(1e-3, float(np.mean(y)))
    pos_weight = torch.tensor((1 - pos) / pos, device=device)
    best, best_state, hist = -1.0, None, []
    for step in range(1, steps + 1):
        model.train()
        idx = rng.integers(0, len(x), batch)
        xb, yb = x[idx], y[idx]
        if extra is not None:
            m = rng.random(batch) < p_extra
            if m.any():
                j = rng.integers(0, len(extra[0]), int(m.sum()))
                xb = xb.copy(); yb = yb.copy()
                xb[m], yb[m] = extra[0][j], extra[1][j]
        xt = torch.tensor(xb, dtype=torch.float32)
        if augment is not None:
            xa = augment(xt, gen)
            mask = torch.rand(batch, 1, generator=gen) < p_aug
            xt = torch.where(mask, xa, xt)
        xt = xt.to(device)[:, None]
        yt = torch.tensor(yb, dtype=torch.float32, device=device)
        loss = F.binary_cross_entropy_with_logits(model(xt)[:, 0], yt, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps:
            a = auroc(y_val, predict(model, x_val, device))
            if x_val_aug is not None:
                a = 0.5 * (a + auroc(y_val, predict(model, x_val_aug, device)))
            hist.append({"step": step, "val_auroc": a, "loss": loss.item()})
            if a > best:
                best = a
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, {"best_val_auroc": best, "history": hist}
