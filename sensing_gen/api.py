"""Public API: use the sensor model, the ECG generator and the electrode surrogate in your own pipeline.

    import numpy as np
    from sensing_gen import api

    x = np.load("my_ecg.npy")                       # (N, L) single-lead ECG in millivolts, 100 Hz
    noisy = api.SensorAugmenter(seed=0)(x)          # physics sensor/artifact model, same units

    gen = api.ECGGenerator.from_pretrained()        # conditional diffusion model (PTB-XL lead II)
    ecg_mv, labels = gen.sample(64, label="abnormal", hr_bpm=80)

    sur = api.ElectrodeSurrogate.from_pretrained()  # neural surrogate of the FE electrode solver
    z, sigma = sur.impedance(w_el_mm=10, lift=0.2)  # complex |Z| spectrum (ohm) + uncertainty

All signals are single-lead ECG at 100 Hz, 10-s windows (L = 1000) for the generator. See
docs/MODEL_CARD.md for training data, intended use and limitations.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).resolve().parent / "weights"
MV_PER_UNIT = 0.14618287980556488     # PTB-XL lead II training std (mV) = 1 model unit
FS = 100.0
LABELS = {"norm": 0, "normal": 0, "abnormal": 1, "abn": 1, 0: 0, 1: 1}


def _resample(x: np.ndarray, fs: float) -> np.ndarray:
    if abs(fs - FS) < 1e-9:
        return x
    from fractions import Fraction
    from scipy import signal as sps
    f = Fraction(FS / fs).limit_denominator(1000)
    return sps.resample_poly(x, f.numerator, f.denominator, axis=-1)


class SensorAugmenter:
    """Physics-based electrode-skin sensor model as a numpy-in, numpy-out augmentation.

    Adds motion-coupled skin-stretch potential, impedance-modulated gain, mains pickup (60 Hz aliased),
    burst EMG, baseline wander, contact loss and ADC effects. Input in millivolts at `fs` Hz; output
    in millivolts at 100 Hz.
    """

    def __init__(self, seed: int | None = None, snr_db: tuple[float, float] | None = None,
                 multiphysics_priors: str | None = None):
        import torch
        from .physics.sensor import ElectrodeSkinSensor, SensorConfig
        cfg = SensorConfig.from_multiphysics(multiphysics_priors) if multiphysics_priors else SensorConfig()
        if snr_db is not None:
            cfg = replace(cfg, snr_db=tuple(snr_db))
        self.sensor = ElectrodeSkinSensor(cfg)
        self.gen = torch.Generator().manual_seed(seed) if seed is not None else None

    def __call__(self, x_mv: np.ndarray, fs: float = FS, return_parts: bool = False):
        import torch
        x = np.atleast_2d(np.asarray(x_mv, dtype=np.float64))
        x = _resample(x, fs) / MV_PER_UNIT
        out = self.sensor(torch.tensor(x, dtype=torch.float32), self.gen, return_parts=return_parts)
        if return_parts:
            y, parts = out
            return y.numpy() * MV_PER_UNIT, {k: v.numpy() * MV_PER_UNIT for k, v in parts.items()}
        return out.numpy() * MV_PER_UNIT


class ECGGenerator:
    """Conditional 1-D diffusion model for 10-s single-lead ECG (label: NORM / abnormal, heart rate)."""

    def __init__(self, diffusion, device: str = "cpu", sampler: dict | None = None):
        self.diff, self.device = diffusion.eval(), device
        self.sampler = sampler or {"spacing": "leading", "threshold": "dynamic", "w": 1.0}

    @classmethod
    def from_pretrained(cls, path: str | Path | None = None, device: str = "cpu") -> "ECGGenerator":
        import torch
        from .diffusion.train import make_model
        path = Path(path) if path else WEIGHTS / "ecg_diffusion_ptbxl_lead2.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; train with experiments/run_diffusion.py or pass a path")
        m = make_model().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        return cls(m, device)

    def sample(self, n: int, label="norm", hr_bpm: float | np.ndarray = 70.0, steps: int = 50,
               guidance: float | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Return (n, 1000) ECG in millivolts at 100 Hz and the integer labels (0 = NORM, 1 = abnormal)."""
        from .diffusion.train import generate
        lab = np.full(n, LABELS[label.lower() if isinstance(label, str) else int(label)], dtype=np.int64)
        hr = np.broadcast_to(np.asarray(hr_bpm, dtype=np.float32), (n,)).copy()
        x = generate(self.diff, lab, hr, self.device, steps=steps,
                     w=self.sampler["w"] if guidance is None else guidance, seed=seed,
                     spacing=self.sampler["spacing"], threshold=self.sampler["threshold"])
        return x * MV_PER_UNIT, lab


class ElectrodeSurrogate:
    """Neural surrogate of the FE electrode-skin solver: parameters -> two-electrode impedance spectrum.

    Parameters (defaults = the nominal wet electrode): t_derm_mm, t_fat_mm, sig_derm, sig_fat, sig_mus,
    sig_sc, eps_sc, w_el_mm, s_el_mm, lift. Valid inside the training ranges in surrogate/space.py; use
    the returned sigma (log10|Z| units) to detect extrapolation, or `exact=True` for the FE solver.
    """

    DEFAULTS = {"t_derm_mm": 1.5, "t_fat_mm": 4.0, "sig_derm": 0.20, "sig_fat": 0.04, "sig_mus": 0.35,
                "sig_sc": 2e-5, "eps_sc": 1000.0, "w_el_mm": 10.0, "s_el_mm": 30.0, "lift": 0.0}

    def __init__(self, model=None):
        self.model = model

    @classmethod
    def from_pretrained(cls, path: str | Path | None = None) -> "ElectrodeSurrogate":
        from .surrogate.model import load_surrogate
        path = Path(path) if path else WEIGHTS / "electrode_surrogate.pt"
        return cls(load_surrogate(path))

    @staticmethod
    def frequencies() -> np.ndarray:
        from .surrogate.space import FREQS
        return FREQS.copy()

    def _theta(self, rows) -> np.ndarray:
        from .surrogate.space import NAMES
        return np.array([[r.get(k, self.DEFAULTS[k]) for k in NAMES] for r in rows], dtype=float)

    def impedance(self, exact: bool = False, **params):
        """Complex impedance (ohm) at the 21 frequencies of `frequencies()` and, unless exact, the
        ensemble standard deviation of log10|Z|."""
        from .surrogate.dataset import from_targets, simulate_one
        unknown = set(params) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"unknown parameters {sorted(unknown)}; valid: {sorted(self.DEFAULTS)}")
        th = self._theta([params])
        if exact:
            return simulate_one(th[0]), None
        if self.model is None:
            raise RuntimeError("no surrogate loaded; use ElectrodeSurrogate.from_pretrained()")
        m, s = self.model.predict(th)
        return from_targets(m)[0], s[0, : m.shape[1] // 2]
