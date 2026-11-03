"""Differentiable ECG dynamical model (McSharry, Clifford, Tarassenko & Smith, IEEE TBME 2003).

The original model is a 3-D ODE whose (x, y) state moves on a unit limit cycle (one revolution per
RR interval) while z, the ECG, is pushed by Gaussian "events" at fixed angles (P, Q, R, S, T):

    dz/dt = - sum_i a_i * dtheta_i * exp(-dtheta_i^2 / (2 b_i^2)) - (z - z0(t))
    dtheta_i = (theta(t) - theta_i) mod 2pi,   dtheta/dt = omega(t) = 2pi / RR(t)

On the limit cycle theta(t) is just the integrated instantaneous heart rate, so we drive the phase
directly from an RR process with Mayer-wave (0.1 Hz) and respiratory sinus arrhythmia components.
The z equation is linear, so it is integrated *exactly* for piecewise-constant forcing with an
exponential integrator implemented as an FFT convolution: fully vectorised over the batch and
differentiable w.r.t. every parameter (used for calibration to real beats and by the PINN).
A classical RK4 integrator is kept as a reference implementation and is checked against it in tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

WAVES = ("P", "Q", "R", "S", "T")
DEFAULT_THETA = (-math.pi / 3, -math.pi / 12, 0.0, math.pi / 12, math.pi / 2)
DEFAULT_A = (1.2, -5.0, 30.0, -7.5, 0.75)
DEFAULT_B = (0.25, 0.1, 0.1, 0.1, 0.4)
TWO_PI = 2 * math.pi


def wrap(x: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi)."""
    return torch.remainder(x + math.pi, TWO_PI) - math.pi


@dataclass
class ECGParams:
    theta: torch.Tensor   # (B, K) event angles [rad]
    a: torch.Tensor       # (B, K) event amplitudes
    b: torch.Tensor       # (B, K) event widths [rad]
    offset: torch.Tensor  # (B,) DC offset of the observed signal

    @property
    def batch(self) -> int:
        return self.theta.shape[0]

    def to(self, device) -> "ECGParams":
        return ECGParams(self.theta.to(device), self.a.to(device), self.b.to(device), self.offset.to(device))

    def detach(self) -> "ECGParams":
        return ECGParams(self.theta.detach(), self.a.detach(), self.b.detach(), self.offset.detach())

    def as_vector(self) -> torch.Tensor:
        return torch.cat([self.theta, self.a, self.b, self.offset[:, None]], dim=1)

    @staticmethod
    def from_vector(v: torch.Tensor, k: int = 5) -> "ECGParams":
        return ECGParams(v[:, :k], v[:, k : 2 * k], v[:, 2 * k : 3 * k], v[:, 3 * k])


def default_params(batch: int, gain: float = 1.0, device="cpu") -> ECGParams:
    t = lambda v: torch.tensor(v, dtype=torch.float32, device=device).repeat(batch, 1)
    return ECGParams(t(DEFAULT_THETA), t(DEFAULT_A) * gain, t(DEFAULT_B),
                     torch.zeros(batch, device=device))


def forcing(phase: torch.Tensor, p: ECGParams) -> torch.Tensor:
    """Right-hand side event term F(theta) for phases of shape (B, T). Returns (B, T)."""
    d = wrap(phase[..., None] - p.theta[:, None, :])
    b = p.b[:, None, :]
    return -(p.a[:, None, :] * d * torch.exp(-(d**2) / (2 * b**2))).sum(-1)


def integrate_exp(u: torch.Tensor, dt: float, z_init: torch.Tensor | None = None) -> torch.Tensor:
    """Exact solution of dz/dt = u(t) - z for piecewise-constant u (zero-order hold).

    z[n] = q^n z0 + sum_{m<n} (1-q) q^(n-1-m) u[m],  q = exp(-dt).  Computed by FFT convolution.
    """
    B, T = u.shape
    q = math.exp(-dt)
    k = torch.arange(T, device=u.device, dtype=u.dtype)
    h = (1 - q) * torch.pow(torch.tensor(q, device=u.device, dtype=u.dtype), k)
    n = 1 << (2 * T - 1).bit_length()
    conv = torch.fft.irfft(torch.fft.rfft(u, n=n) * torch.fft.rfft(h, n=n), n=n)[:, :T]
    z = torch.zeros_like(u)
    z[:, 1:] = conv[:, :-1]
    if z_init is not None:
        z = z + z_init[:, None] * torch.pow(torch.tensor(q, device=u.device, dtype=u.dtype), k)[None]
    return z


def integrate_rk4(phase: torch.Tensor, p: ECGParams, z0: torch.Tensor, dt: float) -> torch.Tensor:
    """Reference RK4 integrator (slow, Python loop). phase must be unwrapped (monotone)."""
    B, T = phase.shape
    z = torch.zeros(B, T, dtype=phase.dtype, device=phase.device)
    for n in range(T - 1):
        ph_mid = 0.5 * (phase[:, n] + phase[:, n + 1])
        zb_mid = 0.5 * (z0[:, n] + z0[:, n + 1])
        f = lambda ph, zz, zb: forcing(ph[:, None], p)[:, 0] - (zz - zb)
        k1 = f(phase[:, n], z[:, n], z0[:, n])
        k2 = f(ph_mid, z[:, n] + 0.5 * dt * k1, zb_mid)
        k3 = f(ph_mid, z[:, n] + 0.5 * dt * k2, zb_mid)
        k4 = f(phase[:, n + 1], z[:, n] + dt * k3, z0[:, n + 1])
        z[:, n + 1] = z[:, n] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


def rr_process(hr: torch.Tensor, rr_std: torch.Tensor, resp_hz: torch.Tensor, n: int, fs: float,
               gen: torch.Generator | None = None, lf_hf: float = 0.5) -> torch.Tensor:
    """RR(t) [s] on a grid of n samples: bimodal spectrum (Mayer 0.1 Hz + RSA at the breathing rate)."""
    B = hr.shape[0]
    dev = hr.device
    f = torch.fft.rfftfreq(n, d=1 / fs).to(dev)
    c1, c2 = 0.01, 0.01
    s1 = torch.exp(-((f[None] - 0.1) ** 2) / (2 * c1**2)) * lf_hf
    s2 = torch.exp(-((f[None] - resp_hz[:, None]) ** 2) / (2 * c2**2))
    amp = torch.sqrt(s1 + s2)
    ph = torch.rand(B, f.numel(), generator=gen).to(dev) * TWO_PI
    proc = torch.fft.irfft(amp * torch.exp(1j * ph), n=n)
    proc = (proc - proc.mean(1, keepdim=True)) / (proc.std(1, keepdim=True) + 1e-8)
    rr = 60.0 / hr[:, None] + rr_std[:, None] * proc
    return rr.clamp(0.3, 2.0)


class ECGSimulator:
    """Batched, differentiable ECG simulator with cardio-respiratory coupling."""

    def __init__(self, fs_out: float = 100.0, fs_int: float = 500.0, warmup_s: float = 3.0):
        assert fs_int % fs_out == 0
        self.fs_out, self.fs_int, self.warmup_s = fs_out, fs_int, warmup_s
        self.dec = int(fs_int // fs_out)

    def simulate(self, p: ECGParams, duration_s: float, hr: torch.Tensor, rr_std: torch.Tensor,
                 resp_hz: torch.Tensor, bw_amp: torch.Tensor | None = None,
                 am_depth: torch.Tensor | None = None, gen: torch.Generator | None = None,
                 af: torch.Tensor | None = None) -> torch.Tensor:
        """Return (B, duration_s * fs_out) ECG. `af` (bool, B) switches to an irregular, P-less rhythm
        with fibrillatory waves (atrial-fibrillation-like)."""
        B, dev = p.batch, p.theta.device
        dt = 1.0 / self.fs_int
        n = int(round((duration_s + self.warmup_s) * self.fs_int))
        t = torch.arange(n, device=dev, dtype=torch.float32) * dt
        rr = rr_process(hr, rr_std, resp_hz, n, self.fs_int, gen)
        if af is not None and bool(af.any()):
            rr_af = self._af_rr(hr, n, gen).to(dev)
            rr = torch.where(af[:, None], rr_af, rr)
        phase = torch.cumsum(TWO_PI / rr * dt, dim=1) + torch.rand(B, 1, generator=gen).to(dev) * TWO_PI
        params = p
        if af is not None and bool(af.any()):
            a = p.a.clone()
            a[af, 0] = 0.0  # no organised atrial activity -> no P wave
            params = ECGParams(p.theta, a, p.b, p.offset)
        resp_phase = TWO_PI * resp_hz[:, None] * t[None] + torch.rand(B, 1, generator=gen).to(dev) * TWO_PI
        z0 = torch.zeros(B, n, device=dev) if bw_amp is None else bw_amp[:, None] * torch.sin(resp_phase)
        z = integrate_exp(forcing(phase, params) + z0, dt)
        if am_depth is not None:
            z = z * (1 + am_depth[:, None] * torch.sin(resp_phase))
        if af is not None and bool(af.any()):
            ff = torch.empty(B, 1).uniform_(4.0, 7.0, generator=gen).to(dev)
            r_amp = params.a[:, 2:3].abs() * params.b[:, 2:3] ** 2 / TWO_PI  # ~R-wave height scale
            fwave = 0.06 * r_amp * torch.sin(TWO_PI * ff * t[None])
            z = z + fwave * af[:, None].float()
        z = z[:, int(self.warmup_s * self.fs_int) :: self.dec]
        return z[:, : int(round(duration_s * self.fs_out))] + p.offset[:, None]

    def _af_rr(self, hr: torch.Tensor, n: int, gen) -> torch.Tensor:
        B = hr.shape[0]
        out = torch.empty(B, n)
        for i in range(B):
            mean_rr = 60.0 / float(hr[i])
            beats = (torch.randn(64, generator=gen) * 0.22 + 1.0).clamp(0.5, 1.8) * mean_rr
            samples = (beats * self.fs_int).long().clamp(min=1)
            seq = torch.repeat_interleave(beats, samples)
            while seq.numel() < n:
                seq = torch.cat([seq, seq])
            out[i] = seq[:n]
        return out.clamp(0.3, 2.0)

    def simulate_beat(self, p: ECGParams, rr: torch.Tensor, pre_s: float = 0.30,
                      post_s: float = 0.50, history_s: float = 4.0) -> torch.Tensor:
        """Steady-state single beat aligned on the R event (theta = 0 at t = 0).

        Returns (B, (pre+post) * fs_out). Used to calibrate the model to real median beats.
        """
        dev = p.theta.device
        dt = 1.0 / self.fs_int
        n_hist = int(round(history_s * self.fs_int))
        n_post = int(round(post_s * self.fs_int))
        t = (torch.arange(-n_hist, n_post, device=dev, dtype=torch.float32)) * dt
        phase = TWO_PI * t[None] / rr[:, None]
        z = integrate_exp(forcing(phase, p), dt)
        i0 = n_hist - int(round(pre_s * self.fs_int))
        return z[:, i0 :: self.dec] + p.offset[:, None]


def r_wave_gain(target_peak: float = 1.0, rr: float = 0.85) -> float:
    """Scale for DEFAULT_A so that the default beat has an R-peak of `target_peak` (after offset removal)."""
    sim = ECGSimulator()
    beat = sim.simulate_beat(default_params(1), torch.tensor([rr]))[0]
    return float(target_peak / (beat.max() - beat[:5].mean()))
