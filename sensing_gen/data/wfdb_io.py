"""Minimal, dependency-free reader for PhysioNet WFDB records.

Only what this project needs: header parsing plus signal formats 16 (int16 little-endian)
and 212 (packed 12-bit, used by MIT-BIH / NSTDB). Avoids a hard dependency on `wfdb`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import numpy as np

PathLike = Union[str, Path]


@dataclass
class SignalSpec:
    file_name: str
    fmt: int
    gain: float
    baseline: int
    units: str
    description: str
    spf: int = 1            # samples per frame (multi-frequency records)


@dataclass
class Header:
    record_name: str
    n_sig: int
    fs: float
    n_samples: int
    signals: List[SignalSpec] = field(default_factory=list)


def _parse_gain(token: str, adc_zero: int) -> tuple[float, int, str]:
    """Parse a WFDB gain token such as '1000.0(0)/mV', '200/mV' or '200'."""
    units = "mV"
    if "/" in token:
        token, units = token.split("/", 1)
    baseline = adc_zero
    if "(" in token:
        g, b = token.split("(", 1)
        baseline = int(b.rstrip(")"))
        token = g
    gain = float(token) if token else 200.0
    if gain == 0:
        gain = 200.0  # WFDB default
    return gain, baseline, units


def read_header(path: PathLike) -> Header:
    path = Path(path)
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    rec = lines[0].split()
    name = rec[0].split("/")[0]
    n_sig = int(rec[1])
    fs = float(rec[2].split("/")[0]) if len(rec) > 2 else 250.0
    n_samples = int(rec[3]) if len(rec) > 3 else 0
    sigs: List[SignalSpec] = []
    for ln in lines[1 : 1 + n_sig]:
        tok = ln.split()
        ftok = tok[1].split(":")[0].split("+")[0]
        fmt = int(ftok.split("x")[0])
        spf = int(ftok.split("x")[1]) if "x" in ftok else 1
        adc_zero = int(tok[4]) if len(tok) > 4 else 0
        gain, baseline, units = _parse_gain(tok[2] if len(tok) > 2 else "200", adc_zero)
        desc = " ".join(tok[8:]) if len(tok) > 8 else f"sig{len(sigs)}"
        sigs.append(SignalSpec(tok[0], fmt, gain, baseline, units, desc, spf))
    return Header(name, n_sig, fs, n_samples, sigs)


def decode_212(raw: np.ndarray, n_values: int) -> np.ndarray:
    """Decode WFDB format 212 (two 12-bit two's-complement samples in 3 bytes)."""
    raw = raw.astype(np.int32)
    n_triplets = len(raw) // 3
    b = raw[: n_triplets * 3].reshape(-1, 3)
    s0 = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
    s1 = b[:, 2] | ((b[:, 1] & 0xF0) << 4)
    out = np.empty(n_triplets * 2, dtype=np.int32)
    out[0::2] = s0
    out[1::2] = s1
    out[out > 2047] -= 4096
    return out[:n_values]


def encode_212(values: np.ndarray) -> bytes:
    """Inverse of decode_212 (used in tests)."""
    v = np.asarray(values, dtype=np.int32).copy()
    if len(v) % 2:
        v = np.append(v, 0)
    v &= 0xFFF
    s0, s1 = v[0::2], v[1::2]
    b0 = s0 & 0xFF
    b1 = ((s0 >> 8) & 0x0F) | (((s1 >> 8) & 0x0F) << 4)
    b2 = s1 & 0xFF
    return np.stack([b0, b1, b2], axis=1).astype(np.uint8).tobytes()


def read_record(record_path: PathLike) -> tuple[np.ndarray, Header]:
    """Read a record (path without extension). Returns (signal[n_samples, n_sig] in physical units, header)."""
    record_path = Path(record_path)
    hdr = read_header(record_path.with_suffix(".hea"))
    if not hdr.signals:
        raise ValueError(f"No signals in {record_path}")
    fmt = hdr.signals[0].fmt
    if any(s.fmt != fmt for s in hdr.signals):
        raise ValueError("Mixed signal formats are not supported")
    dat = record_path.parent / hdr.signals[0].file_name
    raw = np.fromfile(dat, dtype=np.uint8)
    if fmt == 16:
        digital = raw.view("<i2").astype(np.int32)
    elif fmt == 212:
        n_vals = hdr.n_samples * hdr.n_sig if hdr.n_samples else (len(raw) // 3) * 2
        digital = decode_212(raw, n_vals)
    else:
        raise NotImplementedError(f"WFDB format {fmt} not supported")
    n = len(digital) // hdr.n_sig
    digital = digital[: n * hdr.n_sig].reshape(n, hdr.n_sig)
    gains = np.array([s.gain for s in hdr.signals], dtype=np.float64)
    base = np.array([s.baseline for s in hdr.signals], dtype=np.float64)
    phys = (digital - base) / gains
    return phys.astype(np.float32), hdr


def read_record_multi(record_path: PathLike) -> tuple[dict, Header]:
    """Read a (possibly multi-frequency) record. Returns ({description: (signal, fs)}, header).

    Supports formats 16 and 212 with per-signal samples-per-frame ("16x31" style tokens); the
    base sampling frequency in the header is the *frame* rate.
    """
    record_path = Path(record_path)
    hdr = read_header(record_path.with_suffix(".hea"))
    fmt = hdr.signals[0].fmt
    if any(s.fmt != fmt for s in hdr.signals):
        raise ValueError("Mixed signal formats are not supported")
    raw = np.fromfile(record_path.parent / hdr.signals[0].file_name, dtype=np.uint8)
    if fmt == 16:
        digital = raw[: len(raw) // 2 * 2].view("<i2").astype(np.int32)
    elif fmt == 212:
        digital = decode_212(raw, (len(raw) // 3) * 2)
    else:
        raise NotImplementedError(f"WFDB format {fmt} not supported")
    spf = np.array([s.spf for s in hdr.signals])
    per_frame = int(spf.sum())
    n_frames = len(digital) // per_frame
    if hdr.n_samples:
        n_frames = min(n_frames, hdr.n_samples)
    frames = digital[: n_frames * per_frame].reshape(n_frames, per_frame)
    out, col = {}, 0
    for s_, k in zip(hdr.signals, spf):
        x = frames[:, col : col + k].reshape(-1).astype(np.float64)
        col += k
        name = s_.description.strip() or s_.file_name
        out[name] = (((x - s_.baseline) / s_.gain).astype(np.float32), hdr.fs * k)
    return out, hdr
