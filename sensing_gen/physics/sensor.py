"""Physics-based model of a dry/wet-electrode ECG acquisition chain in a moving vehicle.

Lumped-element view of one electrode pair:

    ECG source --[ Z_e(t) electrode-skin impedance ]--+-- amplifier input Z_in (AC-coupled, tau = R C)
                     ^                                 |
           skin-stretch potential  dV(t) = k_e * d(t)  +-- common-mode mains (60 Hz) converted to
           (motion of the electrode relative to skin)      differential mode by the impedance mismatch

* Mechanical excitation: road / body-motion events form a Poisson process that drives a damped
  2nd-order (mass-spring-damper) seat/torso response d(t) with natural frequency 1.5-4 Hz.
* Electrode motion artifact: skin-stretch potential proportional to d(t), high-passed by the
  amplifier AC coupling; Z_e(t) = Z_e0 (1 + kappa d(t)) also modulates the ECG gain through the
  voltage divider Z_in / (Z_in + Z_e) and the mains interference amplitude (|dZ| / Z_in).
* Baseline wander: respiration-locked plus slow drift (0.05-0.5 Hz).
* EMG: band-limited Gaussian noise gated by a smooth burst envelope (gripping the wheel, talking).
* Contact loss: electrode lift-off -> exponential drift toward the amplifier rail (saturation).
* ADC: finite resolution and clipping.

Mains interference at 60 Hz aliases to |60 - 100| = 40 Hz at fs = 100 Hz, which is simulated directly.
All filters are FFT convolutions, so the model runs batched on CPU or GPU inside the training loop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

TWO_PI = 2 * math.pi


def _fft_conv(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Causal linear convolution of each row of x (B, T) with kernel(s) h (T_h,) or (B, T_h)."""
    T = x.shape[-1]
    n = 1 << (T + h.shape[-1] - 1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(x, n=n) * torch.fft.rfft(h, n=n), n=n)
    return y[..., :T]


def _highpass(x: torch.Tensor, fc: torch.Tensor, fs: float) -> torch.Tensor:
    """First-order RC high-pass (AC coupling) with per-row cut-off fc (B,)."""
    T = x.shape[-1]
    t = torch.arange(T, dtype=x.dtype, device=x.device) / fs
    tau = 1.0 / (TWO_PI * fc[:, None])
    h = torch.exp(-t[None] / tau) / (tau * fs)  # unit-DC-gain low-pass impulse response
    return x - _fft_conv(x, h)


@dataclass
class SensorConfig:
    fs: float = 100.0
    snr_db: tuple[float, float] = (-3.0, 18.0)   # SNR of additive artifacts w.r.t. the clean ECG
    p_baseline: float = 0.9
    p_motion: float = 0.7
    p_emg: float = 0.6
    p_mains: float = 0.4
    p_contact_loss: float = 0.1
    adc_bits: int = 12
    adc_range: float = 10.0                       # +- full scale in normalised units
    kappa_range: tuple[float, float] = (0.0, 0.5)         # impedance sensitivity to motion
    ze_ratio_range: tuple[float, float] = (0.001, 0.05)   # Z_e0 / Z_in

    @staticmethod
    def from_multiphysics(path) -> "SensorConfig":
        """Priors set from the electrode-skin field model (experiments/run_multiphysics.py):
        the kappa and Z_e0/Z_in ranges span what the FV model predicts for wet and dry electrodes
        with 10-100 MOhm front ends (+25 % margin on kappa)."""
        import json
        c = json.loads(open(path).read())["sensor_prior_check"]
        k_hi = 1.25 * c["kappa_multiphysics_range"][1]
        lo, hi = c["ze_over_zin_multiphysics_range"]
        return SensorConfig(kappa_range=(0.0, float(k_hi)), ze_ratio_range=(float(lo), float(hi)))


class ElectrodeSkinSensor:
    def __init__(self, cfg: SensorConfig | None = None):
        self.cfg = cfg or SensorConfig()

    # ---- individual mechanisms -------------------------------------------------------------
    def mechanical_response(self, B: int, T: int, gen) -> torch.Tensor:
        fs = self.cfg.fs
        rate = torch.empty(B, 1).uniform_(0.1, 0.8, generator=gen)          # events / s
        events = (torch.rand(B, T, generator=gen) < rate / fs).float()
        events = events * torch.randn(B, T, generator=gen)                   # signed impulse sizes
        fn = torch.empty(B, 1).uniform_(1.5, 4.0, generator=gen)             # natural freq [Hz]
        zeta = torch.empty(B, 1).uniform_(0.15, 0.5, generator=gen)          # damping ratio
        t = torch.arange(int(3 * fs)).float()[None] / fs
        wn = TWO_PI * fn
        wd = wn * torch.sqrt(1 - zeta**2)
        h = torch.exp(-zeta * wn * t) * torch.sin(wd * t)                    # impulse response
        h = h / (h.abs().sum(1, keepdim=True) + 1e-8)
        return _fft_conv(events, h)

    def baseline(self, B: int, T: int, gen) -> torch.Tensor:
        fs = self.cfg.fs
        t = torch.arange(T).float()[None] / fs
        out = torch.zeros(B, T)
        for _ in range(3):
            f = torch.empty(B, 1).uniform_(0.05, 0.5, generator=gen)
            ph = torch.rand(B, 1, generator=gen) * TWO_PI
            out += torch.rand(B, 1, generator=gen) * torch.sin(TWO_PI * f * t + ph)
        return out

    def emg(self, B: int, T: int, gen) -> torch.Tensor:
        w = torch.randn(B, T + 1, generator=gen)
        hf = w[:, 1:] - w[:, :-1]                                            # emphasise >20 Hz
        env = torch.randn(B, T, generator=gen)
        k = torch.hann_window(int(0.8 * self.cfg.fs))
        env = torch.relu(_fft_conv(env, k / k.sum()) * 3.0)                  # smooth bursts
        return hf * env

    # ---- full chain ------------------------------------------------------------------------
    @torch.no_grad()
    def __call__(self, x: torch.Tensor, gen: torch.Generator | None = None,
                 snr_db: torch.Tensor | None = None, return_parts: bool = False):
        cfg = self.cfg
        dev = x.device
        x = x.detach().float().cpu()
        B, T = x.shape
        u = lambda lo, hi: torch.empty(B, 1).uniform_(lo, hi, generator=gen)
        on = lambda p: (torch.rand(B, 1, generator=gen) < p).float()

        d = self.mechanical_response(B, T, gen)
        d = d / (d.std(1, keepdim=True) + 1e-8)

        # electrode-skin interface
        kappa = u(*cfg.kappa_range)                            # impedance sensitivity to motion
        if tuple(cfg.ze_ratio_range) == (0.001, 0.05):        # default prior: uniform
            ze_over_zin = u(0.001, 0.05)                       # Z_e0 / Z_in
        else:                                                  # wide (multiphysics) prior: log-uniform
            ze_over_zin = torch.exp(u(math.log(cfg.ze_ratio_range[0]), math.log(cfg.ze_ratio_range[1])))
        ze_ratio = ze_over_zin * (1 + kappa * d).clamp(min=0.05)
        gain = 1.0 / (1.0 + ze_ratio)                          # voltage divider
        motion = _highpass(d, u(0.05, 0.7)[:, 0], cfg.fs) * on(cfg.p_motion)
        mismatch = (ze_ratio - ze_over_zin).abs() + u(0.0, 0.01)
        t = torch.arange(T).float()[None] / cfg.fs
        f_alias = abs(60.0 - cfg.fs) if cfg.fs < 120 else 60.0
        mains = mismatch * torch.sin(TWO_PI * f_alias * t + u(0, TWO_PI)) * on(cfg.p_mains)
        bw = self.baseline(B, T, gen) * on(cfg.p_baseline)
        emg = self.emg(B, T, gen) * on(cfg.p_emg)

        parts = {"baseline": bw, "motion": motion, "emg": emg, "mains": mains}
        # random mixing weights, then scale the sum to the requested SNR
        w = torch.rand(B, 4, generator=gen) + 0.05
        noise = torch.zeros(B, T)
        for i, v in enumerate(parts.values()):
            noise += w[:, i : i + 1] * v / (v.std(1, keepdim=True) + 1e-8)
        if snr_db is None:
            snr_db = torch.empty(B).uniform_(*cfg.snr_db, generator=gen)
        ps = (x**2).mean(1, keepdim=True)
        pn = (noise**2).mean(1, keepdim=True) + 1e-12
        noise = noise * torch.sqrt(ps / (pn * 10 ** (snr_db.cpu()[:, None] / 10)))
        y = gain * x + noise

        # contact loss: electrode lift-off drifts to the rail
        lost = on(cfg.p_contact_loss)[:, 0] > 0
        for i in torch.nonzero(lost).flatten().tolist():
            L = int(torch.empty(1).uniform_(0.2, 1.5, generator=gen).item() * cfg.fs)
            s = int(torch.randint(0, max(1, T - L), (1,), generator=gen).item())
            rail = cfg.adc_range * (1 if torch.rand(1, generator=gen).item() > 0.5 else -1)
            ramp = 1 - torch.exp(-torch.arange(L).float() / (0.05 * cfg.fs))
            y[i, s : s + L] = y[i, s : s + L] * (1 - ramp) + rail * ramp

        # ADC
        step = 2 * cfg.adc_range / (2**cfg.adc_bits)
        y = torch.clamp(torch.round(y / step) * step, -cfg.adc_range, cfg.adc_range)
        y = y.to(dev)
        if return_parts:
            return y, {k: v.to(dev) for k, v in parts.items()}
        return y
