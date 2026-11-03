"""Shared experiment configuration: profiles, device selection, paths, IO."""
from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("SG_DATA", ROOT / "data" / "sensing_gen"))
OUT = Path(os.environ.get("SG_OUT", ROOT / "outputs" / "sensing_gen"))


@dataclass
class Profile:
    name: str
    calib_steps: int
    pinn_records: int
    pinn_steps: int
    diff_steps_full: int
    diff_steps_low: int
    diff_steps_sim: int
    n_synth: int
    clf_steps: int
    seeds: tuple
    max_train: int | None = None   # cap on train records (smoke test only)


PROFILES = {
    "smoke": Profile("smoke", calib_steps=30, pinn_records=3, pinn_steps=60, diff_steps_full=40,
                     diff_steps_low=20, diff_steps_sim=20, n_synth=128, clf_steps=40, seeds=(0,),
                     max_train=400),
    "full": Profile("full", calib_steps=300, pinn_records=16, pinn_steps=2000,
                    diff_steps_full=8000, diff_steps_low=4000, diff_steps_sim=4000, n_synth=4000,
                    clf_steps=1200, seeds=(0, 1, 2)),
}


def get_profile() -> Profile:
    return PROFILES[os.environ.get("SG_PROFILE", "full")]


def out_dir(profile: Profile) -> Path:
    d = OUT if profile.name == "full" else OUT / profile.name
    d.mkdir(parents=True, exist_ok=True)
    return d


def device() -> str:
    import torch
    if os.environ.get("SG_DEVICE"):
        return os.environ["SG_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def env_info() -> dict:
    try:
        import torch
        tv, dev = torch.__version__, device()
    except ImportError:  # the multiphysics step is numpy/scipy only
        tv, dev = None, "cpu"
    return {"python": platform.python_version(), "torch": tv, "device": dev,
            "machine": platform.machine(), "platform": platform.platform(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def save_json(obj, path: Path) -> None:
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if hasattr(o, "__dataclass_fields__"):
            return asdict(o)
        raise TypeError(type(o))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=conv))


def low_data_indices(y: np.ndarray, frac: float, seed: int = 0) -> np.ndarray:
    """Stratified random subset of the training set."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        pool = np.nonzero(y == c)[0]
        k = max(2, int(round(frac * len(pool))))
        idx.append(rng.choice(pool, k, replace=False))
    return np.sort(np.concatenate(idx))


def load_splits(profile: Profile):
    from ..data.datasets import build_ptbxl
    ds = build_ptbxl(DATA / "ptbxl")
    train, val, test = ds.split(range(1, 9)), ds.split([9]), ds.split([10])
    if profile.max_train:
        train = train.subset(np.arange(min(len(train), profile.max_train)))
        val = val.subset(np.arange(min(len(val), 200)))
        test = test.subset(np.arange(min(len(test), 200)))
    return ds, train, val, test
