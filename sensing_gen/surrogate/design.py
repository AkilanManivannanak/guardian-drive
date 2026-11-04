"""Robust electrode design through the differentiable surrogate, verified with the FE solver.

Design variables: electrode width w and centre-to-centre spacing s of a two-electrode patch.
Figure of merit (per tissue/contact state): a bipolar ECG amplitude grows ~linearly with spacing while
the thermal-noise voltage grows with sqrt(Re Z), so we maximise

    J(w, s) = E_{tissue, lift-off} [ ln s - 0.5 ln Re Z(10 Hz; w, s, tissue, lift) ]

(+1 in J = +8.7 dB SNR) subject to the patch-length budget s + w <= L and an inter-electrode gap
s - w >= g. The expectation runs over the tissue/contact population of the surrogate's parameter space
(lift-off ~ U(0, 0.4) on one electrode), i.e. a robust design, not a nominal-tissue one.
"""
from __future__ import annotations

import numpy as np


from .space import FREQS, NAMES, from_unit, sample, to_unit

IW, IS, IL = NAMES.index("w_el_mm"), NAMES.index("s_el_mm"), NAMES.index("lift")
F10 = int(np.argmin(np.abs(FREQS - 10.0)))
NOMINAL = np.array([1.5, 4.0, 0.20, 0.04, 0.35, 2e-5, 1000.0, 10.0, 30.0, 0.0])


def population(n: int, seed: int = 123, nominal: bool = False) -> np.ndarray:
    if nominal:
        return np.repeat(NOMINAL[None], n, 0)
    return sample(n, seed=seed)


def project(w: float, s: float, L: float, gap: float, wb=(5.0, 15.0), sb=(20.0, 40.0)):
    """Euclidean projection onto {wb, sb box} ∩ {s + w <= L} ∩ {s - w >= gap} by alternating projections."""
    for _ in range(50):
        w, s = float(np.clip(w, *wb)), float(np.clip(s, *sb))
        if s + w > L:
            d = (s + w - L) / 2; w, s = w - d, s - d
        if s - w < gap:
            d = (gap - (s - w)) / 2; w, s = w - d, s + d
    return float(np.clip(w, *wb)), float(np.clip(s, *sb))


def j_from_logz_phase(logz10: np.ndarray, ph10: np.ndarray, s_mm) -> np.ndarray:
    rez = 10 ** logz10 * np.cos(ph10)
    return np.log(np.asarray(s_mm) * 1e-3) - 0.5 * np.log(rez)


def surrogate_objective(model, pop: np.ndarray, w, s, device="cpu"):
    """Differentiable J(w, s) averaged over the population (w, s are torch scalars in mm)."""
    import torch
    u = torch.tensor(to_unit(pop), dtype=torch.float32, device=device)
    u = u.clone()
    u[:, IW] = (w - 5.0) / 10.0
    u[:, IS] = (s - 20.0) / 20.0
    m, _, _ = model(u)
    n = m.shape[1] // 2
    rez = 10 ** m[:, F10] * torch.cos(m[:, n + F10])
    return (torch.log(s * 1e-3) - 0.5 * torch.log(rez)).mean()


def optimise(model, pop, L=35.0, gap=5.0, starts=((6.0, 22.0), (14.0, 20.0), (8.0, 26.0)), steps=300,
             lr=0.2, device="cpu"):
    import torch
    best = None
    traces = []
    for w0, s0 in starts:
        w0, s0 = project(w0, s0, L, gap)
        w = torch.tensor(w0, requires_grad=True, device=device)
        s = torch.tensor(s0, requires_grad=True, device=device)
        opt = torch.optim.Adam([w, s], lr=lr)
        tr = []
        for _ in range(steps):
            j = surrogate_objective(model, pop, w, s, device)
            opt.zero_grad(); (-j).backward(); opt.step()
            with torch.no_grad():
                pw, ps = project(w.item(), s.item(), L, gap)
                w.fill_(pw); s.fill_(ps)
            tr.append((w.item(), s.item(), j.item()))
        traces.append(tr)
        if best is None or tr[-1][2] > best[2]:
            best = tr[-1]
    return {"w_mm": best[0], "s_mm": best[1], "J_surrogate": best[2], "traces": traces}


def fe_objective(pop: np.ndarray, w: float, s: float, workers: int | None = None) -> np.ndarray:
    """Per-sample J evaluated with the finite-element solver at 10 Hz."""
    from .dataset import simulate_10hz
    th = pop.copy(); th[:, IW] = w; th[:, IS] = s
    Z = simulate_10hz(th, workers)
    return np.log(s * 1e-3) - 0.5 * np.log(Z.real)
