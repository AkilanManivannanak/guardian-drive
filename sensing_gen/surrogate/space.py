"""Parameter space of the electrode-skin surrogate and the mapping to simulator configurations."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..multiphysics.electrode_skin_fv import ElectrodeConfig, Layer, TissueModel

# name, low, high, log-scale, unit note
PARAMS = [
    ("t_derm_mm", 1.0, 2.5, False),
    ("t_fat_mm", 1.0, 10.0, False),
    ("sig_derm", 0.10, 0.40, False),
    ("sig_fat", 0.02, 0.08, False),
    ("sig_mus", 0.20, 0.50, False),
    ("sig_sc", 5e-6, 1e-4, True),      # effective (hydrated) stratum-corneum conductivity, S/m
    ("eps_sc", 300.0, 3000.0, True),
    ("w_el_mm", 5.0, 15.0, False),
    ("s_el_mm", 20.0, 40.0, False),     # centre-to-centre spacing
    ("lift", 0.0, 0.4, False),          # fraction of electrode A that has lost contact
]
NAMES = [p[0] for p in PARAMS]
LO = np.array([p[1] for p in PARAMS])
HI = np.array([p[2] for p in PARAMS])
LOG = np.array([p[3] for p in PARAMS])
FREQS = np.logspace(0, 4, 21)
DEPTH_MM = 30.0


def to_unit(theta: np.ndarray) -> np.ndarray:
    """Physical parameters -> [0, 1]^d (log-uniform where flagged)."""
    t = np.asarray(theta, float)
    lo, hi = np.where(LOG, np.log(np.where(LOG, LO, 1.0)), LO), np.where(LOG, np.log(np.where(LOG, HI, 1.0)), HI)
    v = np.where(LOG, np.log(np.maximum(t, 1e-30)), t)
    return (v - lo) / (hi - lo)


def from_unit(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, float)
    lo, hi = np.where(LOG, np.log(np.where(LOG, LO, 1.0)), LO), np.where(LOG, np.log(np.where(LOG, HI, 1.0)), HI)
    v = lo + u * (hi - lo)
    return np.where(LOG, np.exp(v), v)


def sample(n: int, seed: int = 0, expand: float = 0.0) -> np.ndarray:
    """Latin-hypercube sample of n parameter vectors. `expand` > 0 widens every range by that fraction
    on both sides (used for the out-of-distribution test); values are clipped to physical limits."""
    from scipy.stats import qmc
    u = qmc.LatinHypercube(d=len(PARAMS), seed=seed).random(n)
    u = -expand + u * (1 + 2 * expand)
    th = from_unit(u)
    th[:, NAMES.index("lift")] = np.clip(th[:, NAMES.index("lift")], 0.0, 0.6)
    th[:, NAMES.index("t_fat_mm")] = np.clip(th[:, NAMES.index("t_fat_mm")], 0.5, 15.0)
    th[:, NAMES.index("t_derm_mm")] = np.clip(th[:, NAMES.index("t_derm_mm")], 0.6, 3.5)
    return th


def to_configs(theta) -> tuple[TissueModel, ElectrodeConfig, float]:
    p = dict(zip(NAMES, np.asarray(theta, float)))
    td, tf = p["t_derm_mm"] * 1e-3, p["t_fat_mm"] * 1e-3
    tm = TissueModel(layers=[Layer("epidermis+dermis", td, p["sig_derm"], 1.0e4),
                             Layer("fat", tf, p["sig_fat"], 1.0e4),
                             Layer("muscle", DEPTH_MM * 1e-3 - td - tf, p["sig_mus"], 1.0e5)])
    tm = replace(tm, sc_sigma=p["sig_sc"], sc_hydration_gain=1.0, sc_eps_r=p["eps_sc"])
    ec = ElectrodeConfig(kind="wet", width=p["w_el_mm"] * 1e-3, spacing=p["s_el_mm"] * 1e-3)
    return tm, ec, float(p["lift"])


EPS0 = 8.854e-12
SC_T = 20e-6       # stratum-corneum thickness used by the simulator (m)
L_OOP = 10e-3      # out-of-plane electrode length (m)


def unit_to_phys_torch(u):
    """Torch version of from_unit (differentiable)."""
    import torch
    lo = torch.tensor(np.where(LOG, np.log(np.where(LOG, LO, 1.0)), LO), dtype=u.dtype, device=u.device)
    hi = torch.tensor(np.where(LOG, np.log(np.where(LOG, HI, 1.0)), HI), dtype=u.dtype, device=u.device)
    v = lo + u * (hi - lo)
    lg = torch.tensor(LOG, device=u.device)
    return torch.where(lg, v.exp(), v)


def contact_prior(theta, xp=np):
    """Analytic contact-only impedance of the two electrodes in series (no tissue):
    Z_c = (1/w_A + 1/w_B) / (L * Y),  Y = (sigma_sc + j w eps0 eps_sc) / t_sc,  w_A = w (1 - lift).
    Returns targets [log10|Z_c| (21), angle(Z_c) (21)]; works for numpy arrays and torch tensors."""
    f = FREQS if xp is np else xp.tensor(FREQS, dtype=theta.dtype, device=theta.device)
    sig, eps = theta[:, NAMES.index("sig_sc"), None], theta[:, NAMES.index("eps_sc"), None]
    w = theta[:, NAMES.index("w_el_mm"), None] * 1e-3
    lift = theta[:, NAMES.index("lift"), None]
    geo = (1.0 / (w * (1 - lift)) + 1.0 / w) / L_OOP
    b = 2 * np.pi * f[None] * EPS0 * eps
    log_absY = 0.5 * xp.log10(sig ** 2 + b ** 2) - np.log10(SC_T)
    logz = xp.log10(geo) - log_absY
    ph = -xp.arctan2(b, sig * xp.ones_like(b)) if xp is np else -xp.atan2(b, sig.expand_as(b))
    return xp.cat([logz, ph], -1) if xp is not np else np.concatenate([logz, ph], -1)
