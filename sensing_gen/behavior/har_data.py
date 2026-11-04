"""UCI HAR (Anguita et al., 2013): 30 subjects, waist-worn smartphone, 50 Hz, 2.56-s windows (128
samples, 50 % overlap), 6 activities. We use the raw inertial signals: total acceleration (g, includes
gravity) and body angular rate (rad/s), x/y/z, so a rotation of the phone acts on them exactly.
Licence: CC BY 4.0. https://archive.ics.uci.edu/dataset/240
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ACTIVITIES = ["walking", "upstairs", "downstairs", "sitting", "standing", "lying"]
CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
FS = 50.0


def download(root: str | Path) -> Path:
    """Download and unpack into root/'UCI HAR Dataset' (about 60 MB). Idempotent."""
    root = Path(root)
    tgt = root / "UCI HAR Dataset"
    if (tgt / "train" / "y_train.txt").exists():
        return tgt
    root.mkdir(parents=True, exist_ok=True)
    print(f"[har] downloading {URL}", flush=True)
    blob = urllib.request.urlopen(URL, timeout=300).read()
    outer = zipfile.ZipFile(io.BytesIO(blob))
    inner_name = next((n for n in outer.namelist() if n.lower().endswith(".zip")), None)
    z = zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) if inner_name else outer
    z.extractall(root)
    return tgt


def _split(base: Path, split: str):
    sig = base / split / "Inertial Signals"
    names = [f"total_acc_{a}_{split}.txt" for a in "xyz"] + [f"body_gyro_{a}_{split}.txt" for a in "xyz"]
    x = np.stack([np.loadtxt(sig / n, dtype=np.float32) for n in names], axis=1)      # (N, 6, 128)
    y = np.loadtxt(base / split / f"y_{split}.txt", dtype=np.int64) - 1
    s = np.loadtxt(base / split / f"subject_{split}.txt", dtype=np.int64)
    return x, y, s


def load(root: str | Path, cache: bool = True) -> dict:
    """Returns {'train': (x, y, subject), 'test': (...)}; x is (N, 6, 128) float32 in g and rad/s."""
    root = Path(root)
    c = root / "har_cache.npz"
    if cache and c.exists():
        z = np.load(c)
        return {k: (z[f"{k}_x"], z[f"{k}_y"], z[f"{k}_s"]) for k in ("train", "test")}
    base = download(root)
    d = {k: _split(base, k) for k in ("train", "test")}
    if cache:
        np.savez_compressed(c, **{f"{k}_{n}": v for k, t in d.items() for n, v in zip("xys", t)})
    return d
