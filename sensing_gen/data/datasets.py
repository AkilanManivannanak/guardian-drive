"""PTB-XL (single-lead, wearable-style) and MIT-BIH NSTDB loaders with on-disk caching."""
from __future__ import annotations

import ast
import csv
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal as sps

from .signal_utils import bandpass, heart_rate
from .wfdb_io import read_record

FS = 100.0
WIN = 1000  # 10 s at 100 Hz
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


@dataclass
class ECGDataset:
    x: np.ndarray        # (N, 1000) float32, normalised lead II
    y: np.ndarray        # (N,) int64: 0 = NORM, 1 = abnormal (MI/STTC/CD/HYP)
    multi: np.ndarray    # (N, 5) multi-hot diagnostic superclasses
    fold: np.ndarray     # (N,) PTB-XL strat_fold
    hr: np.ndarray       # (N,) bpm (NaN if undetectable)
    rr_std: np.ndarray   # (N,) s
    scale: float         # mV per normalised unit

    def split(self, folds) -> "ECGDataset":
        m = np.isin(self.fold, list(folds))
        return ECGDataset(self.x[m], self.y[m], self.multi[m], self.fold[m], self.hr[m],
                          self.rr_std[m], self.scale)

    def subset(self, idx) -> "ECGDataset":
        return ECGDataset(self.x[idx], self.y[idx], self.multi[idx], self.fold[idx], self.hr[idx],
                          self.rr_std[idx], self.scale)

    def __len__(self) -> int:
        return len(self.y)


def _diag_map(scp_csv: Path) -> dict[str, str]:
    rows = list(csv.reader(io.StringIO(scp_csv.read_text())))
    head = rows[0]
    i_diag, i_cls = head.index("diagnostic"), head.index("diagnostic_class")
    out = {}
    for r in rows[1:]:
        if len(r) > i_cls and r[i_diag].strip() in ("1", "1.0") and r[i_cls].strip():
            out[r[0].strip()] = r[i_cls].strip()
    return out


def build_ptbxl(root: str | Path, cache: str | Path | None = None, lead: str = "II") -> ECGDataset:
    root = Path(root)
    cache = Path(cache) if cache else root / f"ptbxl_lead{lead}_cache.npz"
    if cache.exists():
        z = np.load(cache)
        return ECGDataset(z["x"], z["y"], z["multi"], z["fold"], z["hr"], z["rr_std"], float(z["scale"]))

    diag = _diag_map(root / "scp_statements.csv")
    rows = list(csv.DictReader(io.StringIO((root / "ptbxl_database.csv").read_text())))
    xs, ys, ms, fs_, hrs, rrs = [], [], [], [], [], []
    for r in rows:
        rec = root / r["filename_lr"]
        if not rec.with_suffix(".dat").exists():
            continue
        codes = ast.literal_eval(r["scp_codes"])
        sup = {diag[c] for c in codes if c in diag}
        if not sup:
            continue
        sig, hdr = read_record(rec)
        names = [s.description.strip().upper() for s in hdr.signals]
        li = names.index(lead.upper()) if lead.upper() in names else 1
        x = sig[:WIN, li]
        if len(x) < WIN or not np.isfinite(x).all():
            continue
        x = bandpass(x, FS)
        m = np.array([c in sup for c in SUPERCLASSES], dtype=np.float32)
        xs.append(x)
        ms.append(m)
        ys.append(int(m[1:].any()))
        fs_.append(int(r["strat_fold"]))
        hr, rs = heart_rate(x, FS)
        hrs.append(hr)
        rrs.append(rs)
    if not xs:
        raise FileNotFoundError(f"No PTB-XL records found under {root}. Run sensing_gen.data.download first.")
    X = np.stack(xs).astype(np.float32)
    fold = np.array(fs_)
    scale = float(np.std(X[fold <= 8]))
    X = np.clip(X / scale, -12, 12).astype(np.float32)
    ds = ECGDataset(X, np.array(ys, dtype=np.int64), np.stack(ms), fold,
                    np.array(hrs, dtype=np.float32), np.array(rrs, dtype=np.float32), scale)
    np.savez_compressed(cache, x=ds.x, y=ds.y, multi=ds.multi, fold=ds.fold, hr=ds.hr,
                        rr_std=ds.rr_std, scale=ds.scale)
    return ds


class NoiseBank:
    """Real sensor noise from MIT-BIH NSTDB, resampled 360 -> 100 Hz.

    Each recording is split in time: the first half is the *train* pool (may be used for
    augmentation), the second half is the *test* pool (only ever used to corrupt test data),
    so no noise sample is shared between training and evaluation.
    """

    KINDS = ("bw", "em", "ma")

    def __init__(self, root: str | Path, fs: float = FS):
        root = Path(root)
        self.pools: dict[str, dict[str, np.ndarray]] = {"train": {}, "test": {}}
        for k in self.KINDS:
            sig, hdr = read_record(root / k)
            up, down = int(fs), int(hdr.fs)
            g = np.gcd(up, down)
            sig = sps.resample_poly(sig, up // g, down // g, axis=0).astype(np.float32)
            sig = sig - sig.mean(0, keepdims=True)
            half_len = sig.shape[0] // 2
            tr = np.concatenate([sig[:half_len, c] for c in range(sig.shape[1])])
            te = np.concatenate([sig[half_len:, c] for c in range(sig.shape[1])])
            self.pools["train"][k] = tr
            self.pools["test"][k] = te

    def sample(self, n: int, length: int, split: str, rng: np.random.Generator,
               kinds=("bw", "em", "ma")) -> np.ndarray:
        """Return (n, length) noise; each row is a random mixture of the requested artifact types."""
        out = np.zeros((n, length), dtype=np.float32)
        for i in range(n):
            w = rng.dirichlet(np.ones(len(kinds)))
            for wk, k in zip(w, kinds):
                pool = self.pools[split][k]
                s = rng.integers(0, len(pool) - length)
                seg = pool[s : s + length]
                out[i] += wk * (seg - seg.mean()) / (seg.std() + 1e-8)
        return out


def corrupt_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add noise rows to clean rows so that each row has exactly the requested SNR."""
    ps = np.mean(clean**2, axis=-1, keepdims=True)
    pn = np.mean(noise**2, axis=-1, keepdims=True) + 1e-12
    k = np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    return (clean + k * noise).astype(np.float32)
