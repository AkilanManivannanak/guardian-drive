"""Inverse physics-informed neural network (PINN) for single-lead ECG.

Given a noisy recording y(t) and the cardiac phase theta(t) (from R-peak detection), we learn

    s_net(t)   a Fourier-feature MLP representing the clean ECG
    w(t)       a low-order Fourier series representing baseline wander (< 0.7 Hz)
    phi        the physical parameters (theta_i, a_i, b_i) of the McSharry ECG model

by minimising   L = mean_n (y_n - s(t_n) - w(t_n))^2  +  lambda * mean_c r(t_c)^2,
with the ODE residual   r(t) = ds/dt - F(theta(t); phi) + s(t)   evaluated by autograd on a dense
collocation grid. The physics term turns an under-determined curve fit into a well-posed inverse
problem: noise that is not explainable by the cardiac dynamics is rejected, and phi is identified.

Many segments are fitted *in parallel*: the MLP weights are batched per segment (einsum), so one
optimiser run handles S independent inverse problems.

Also provided: `fit_diffsim`, the hard-constraint counterpart that fits phi by back-propagating
through the exact ODE solver (no network), used as an ablation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..physics.ecg_ode import (DEFAULT_A, DEFAULT_B, DEFAULT_THETA, ECGParams, forcing,
                               integrate_exp, r_wave_gain)

TWO_PI = 2 * math.pi


def phase_from_rpeaks(rpeaks_s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Unwrapped cardiac phase: 0 at each R peak, +2pi per beat, linear in between, extrapolated."""
    rp = np.asarray(rpeaks_s, dtype=np.float64)
    if len(rp) < 2:
        rr = 0.85
        rp = np.array([0.0, rr])
    rr_med = float(np.median(np.diff(rp)))
    pre = np.arange(rp[0] - rr_med, t.min() - 2 * rr_med, -rr_med)[::-1]
    post = np.arange(rp[-1] + rr_med, t.max() + 2 * rr_med, rr_med)
    knots = np.concatenate([pre, rp, post])
    k0 = len(pre)
    ph = TWO_PI * (np.arange(len(knots)) - k0)
    return np.interp(t, knots, ph)


class BatchedLinear(nn.Module):
    def __init__(self, S: int, d_in: int, d_out: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(S, d_in, d_out) * math.sqrt(2.0 / (d_in + d_out)))
        self.b = nn.Parameter(torch.zeros(S, 1, d_out))

    def forward(self, x):  # x (S, N, d_in)
        return torch.einsum("snd,sde->sne", x, self.w) + self.b


class SegmentNet(nn.Module):
    """S independent Fourier-feature MLPs  s(t), evaluated on (S, N) times."""

    def __init__(self, S: int, duration: float, n_freq: int = 96, f_max: float = 30.0,
                 width: int = 128, depth: int = 3):
        super().__init__()
        freqs = torch.logspace(math.log10(0.3), math.log10(f_max), n_freq)  # Hz
        self.register_buffer("freqs", freqs)
        layers, d = [], 2 * n_freq
        for _ in range(depth):
            layers.append(BatchedLinear(S, d, width))
            d = width
        self.hidden = nn.ModuleList(layers)
        self.out = BatchedLinear(S, width, 1)
        self.duration = duration

    def forward(self, t):  # t (S, N) seconds
        arg = TWO_PI * t[..., None] * self.freqs
        h = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
        for layer in self.hidden:
            h = torch.tanh(layer(h))
        return self.out(h)[..., 0]


class Baseline(nn.Module):
    """Low-frequency Fourier series w(t) per segment (baseline wander, < f_max Hz)."""

    def __init__(self, S: int, duration: float, f_max: float = 0.7):
        super().__init__()
        k = max(1, int(f_max * duration))
        self.register_buffer("f", torch.arange(1, k + 1).float() / duration)
        self.c = nn.Parameter(torch.zeros(S, 2 * k + 1))

    def forward(self, t):
        arg = TWO_PI * t[..., None] * self.f
        basis = torch.cat([torch.ones_like(t)[..., None], torch.sin(arg), torch.cos(arg)], -1)
        return (basis * self.c[:, None, :]).sum(-1)


class PhysicsParams(nn.Module):
    def __init__(self, S: int, amp: torch.Tensor):
        super().__init__()
        g = r_wave_gain()
        self.register_buffer("th0", torch.tensor(DEFAULT_THETA).repeat(S, 1))
        self.register_buffer("lb0", torch.log(torch.tensor(DEFAULT_B)).repeat(S, 1))
        # amplitudes are parameterised relative to each segment's scale (well-conditioned for Adam)
        self.register_buffer("a_scale", (g * amp)[:, None].detach().clone().float())
        self.dth = nn.Parameter(torch.zeros(S, 5))
        self.a_n = nn.Parameter(torch.tensor(DEFAULT_A)[None].repeat(S, 1).to(amp.device))
        self.lb = nn.Parameter(self.lb0.clone())

    @property
    def a(self) -> torch.Tensor:
        return self.a_n * self.a_scale

    def set_a(self, a: torch.Tensor) -> None:
        self.a_n.copy_(a / self.a_scale)

    def params(self) -> ECGParams:
        S = self.a_n.shape[0]
        return ECGParams(self.th0 + self.dth, self.a, torch.exp(self.lb).clamp(0.02, 1.2),
                         torch.zeros(S, device=self.a_n.device))

    def prior(self):
        return (self.dth**2).sum(1) + ((self.lb - self.lb0) ** 2).sum(1)


@dataclass
class FitResult:
    denoised: np.ndarray       # (S, T)
    baseline: np.ndarray       # (S, T)
    params: ECGParams
    history: list


def _phase_tensors(rpeaks, T: int, fs: float, oversample: int):
    S = len(rpeaks)
    dur = T / fs
    t_obs = np.arange(T) / fs
    t_col = np.arange(T * oversample) / (fs * oversample)
    ph_col = np.stack([phase_from_rpeaks(rpeaks[i], t_col) for i in range(S)])
    return dur, t_obs, t_col, ph_col


def simulate_from_phase(p: ECGParams, phase_fn, t_end: float, fs_int: float = 500.0,
                        warmup: float = 3.0) -> tuple[torch.Tensor, np.ndarray]:
    """Exact ODE solution on [0, t_end) for a given (numpy) phase function; returns (S, n), t grid."""
    t = np.arange(-warmup, t_end, 1 / fs_int)
    ph = torch.tensor(np.stack([f(t) for f in phase_fn]), dtype=torch.float32, device=p.a.device)
    z = integrate_exp(forcing(ph, p), 1 / fs_int)
    i0 = int(round(warmup * fs_int))
    return z[:, i0:], t[i0:]


@torch.no_grad()
def _project_linear(phys: "PhysicsParams", base: "Baseline", phase_fn, yt, t_obs, T, fs, fs_int,
                    ridge: float = 1e-4) -> None:
    """Variable projection: signal is linear in the wave amplitudes and baseline coefficients, so for
    the current (theta, b) solve both exactly by ridge least squares and write them back."""
    S, K = phys.a_n.shape
    dec = int(fs_int // fs)
    p = phys.params()
    eye = torch.eye(K, device=yt.device)
    a = (eye[None] * phys.a_scale[:, :, None]).reshape(S * K, K)
    pk = ECGParams(p.theta.repeat_interleave(K, 0), a, p.b.repeat_interleave(K, 0),
                   torch.zeros(S * K, device=yt.device))
    fns = [f for f in phase_fn for _ in range(K)]
    z, _ = simulate_from_phase(pk, fns, T / fs, fs_int)
    Z = z[:, ::dec][:, :T].reshape(S, K, T).transpose(1, 2)                  # (S, T, K)
    arg = TWO_PI * t_obs[..., None] * base.f
    Bw = torch.cat([torch.ones_like(t_obs)[..., None], torch.sin(arg), torch.cos(arg)], -1)
    A = torch.cat([Z, Bw], -1).cpu()
    AtA = A.transpose(1, 2) @ A
    n = A.shape[-1]
    reg = ridge * AtA.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * torch.eye(n)[None]
    sol = torch.linalg.solve(AtA + reg, A.transpose(1, 2) @ yt.cpu()[..., None])[..., 0].to(yt.device)
    phys.a_n.copy_(sol[:, :K])
    base.c.copy_(sol[:, K:])


def fit_diffsim(y: np.ndarray, rpeaks: list, fs: float = 100.0, steps: int = 1500,
                lr: float = 0.03, device: str = "cpu") -> FitResult:
    """Hard-constraint inverse problem: y ~ ODE(phi) + w(t); fit phi and w by differentiable simulation."""
    S, T = y.shape
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    amp = torch.tensor(np.percentile(np.abs(y), 99, axis=1), dtype=torch.float32, device=device)
    phys = PhysicsParams(S, amp).to(device)
    base = Baseline(S, T / fs).to(device)
    t_obs = torch.tensor(np.arange(T) / fs, dtype=torch.float32, device=device)[None].repeat(S, 1)
    phase_fn = [lambda tt, r=r: phase_from_rpeaks(r, tt) for r in rpeaks]
    opt = torch.optim.Adam(list(phys.parameters()) + list(base.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    fs_int = 500.0
    dec = int(fs_int // fs)
    hist = []
    for it in range(steps):
        if it % 250 == 0:
            _project_linear(phys, base, phase_fn, yt, t_obs, T, fs, fs_int)
        z, _ = simulate_from_phase(phys.params(), phase_fn, T / fs, fs_int)
        s = z[:, ::dec][:, :T]
        loss = ((yt - s - base(t_obs)) ** 2).mean(1) + 1e-3 * phys.prior()
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
        sched.step()
        if it % 250 == 0:
            hist.append(float(loss.detach().mean()))
    with torch.no_grad():
        z, _ = simulate_from_phase(phys.params(), phase_fn, T / fs, fs_int)
        s = z[:, ::dec][:, :T]
        w = base(t_obs)
    return FitResult(s.cpu().numpy(), w.cpu().numpy(), phys.params().detach(), hist)


def fit_pinn(y: np.ndarray, rpeaks: list, fs: float = 100.0, steps: int = 3000,
             lam: float = 1.0, oversample: int = 4, init_from_diffsim: bool = True,
             diffsim_steps: int = 800, lr: float = 1e-3, device: str = "cpu",
             diffsim_device: str = "cpu", n_col_batch: int = 1500,
             verbose: bool = False) -> FitResult:
    """Soft-constraint inverse PINN (see module docstring). lam = 0 gives the physics-free NN ablation."""
    S, T = y.shape
    dur, t_obs_np, t_col_np, ph_col_np = _phase_tensors(rpeaks, T, fs, oversample)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    t_obs = torch.tensor(t_obs_np, dtype=torch.float32, device=device)[None].repeat(S, 1)
    t_col = torch.tensor(t_col_np, dtype=torch.float32, device=device)[None].repeat(S, 1)
    ph_col = torch.tensor(ph_col_np, dtype=torch.float32, device=device)

    net = SegmentNet(S, dur).to(device)
    base = Baseline(S, dur).to(device)
    amp = torch.tensor(np.percentile(np.abs(y), 99, axis=1), dtype=torch.float32, device=device)
    phys = PhysicsParams(S, amp).to(device)

    hist = []
    if init_from_diffsim and lam > 0:
        ds = fit_diffsim(y, rpeaks, fs, steps=diffsim_steps, device=diffsim_device)
        with torch.no_grad():
            p = ds.params.to(device)
            phys.dth.copy_(p.theta - phys.th0)
            phys.set_a(p.a)
            phys.lb.copy_(torch.log(p.b))
        # distil the physics solution into the network (fast warm start)
        target = torch.tensor(ds.denoised, dtype=torch.float32, device=device)
        wt = torch.tensor(ds.baseline, dtype=torch.float32, device=device)
        opt0 = torch.optim.Adam(net.parameters(), lr=2e-3)
        for _ in range(600):
            l0 = ((net(t_obs) - target) ** 2).mean()
            opt0.zero_grad(); l0.backward(); opt0.step()
        with torch.no_grad():
            # least-squares baseline coefficients for the diffsim baseline
            arg = TWO_PI * t_obs[..., None] * base.f
            basis = torch.cat([torch.ones_like(t_obs)[..., None], torch.sin(arg), torch.cos(arg)], -1)
            sol = torch.linalg.lstsq(basis.cpu(), wt[..., None].cpu()).solution[..., 0]  # CPU: MPS lacks lstsq
            base.c.copy_(sol.to(device))

    ref = ds.denoised if (init_from_diffsim and lam > 0) else y
    dscale = torch.tensor(np.mean((np.gradient(ref, axis=1) * fs) ** 2, axis=1) + 1e-6,
                          dtype=torch.float32, device=device)
    var_y = yt.var(1).detach() + 1e-6
    opt = torch.optim.Adam([
        {"params": net.parameters(), "lr": lr},
        {"params": base.parameters(), "lr": 5e-3},
        {"params": phys.parameters(), "lr": 5e-3},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for it in range(steps):
        data = ((yt - net(t_obs) - base(t_obs)) ** 2).mean(1)
        loss = data
        if lam > 0:
            # stochastic collocation: a fresh random subset of the dense grid every iteration
            ci = torch.randint(0, t_col.shape[1], (min(n_col_batch, t_col.shape[1]),), device=device)
            tc = t_col[:, ci].clone().requires_grad_(True)
            s = net(tc)
            ds_dt = torch.autograd.grad(s.sum(), tc, create_graph=True)[0]
            res = ds_dt - forcing(ph_col[:, ci], phys.params()) + s
            # residual normalised by a fixed derivative scale and multiplied by the signal power,
            # so lam is dimensionless and comparable across segments
            phys_l = (res**2).mean(1) / dscale * var_y
            loss = data + lam * phys_l + 1e-4 * phys.prior()
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
        sched.step()
        if it % 250 == 0:
            hist.append(float(data.detach().mean()))
            if verbose:
                print(f"  pinn it {it:5d} data {data.mean().item():.4f}")
    with torch.no_grad():
        s = net(t_obs)
        w = base(t_obs)
    return FitResult(s.cpu().numpy(), w.cpu().numpy(), phys.params().detach(), hist)
