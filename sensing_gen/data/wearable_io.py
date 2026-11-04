"""Load ECG recorded on consumer wearables, for testing the sensor model on a real device.

Supported formats (detected automatically):
  * Apple Watch ECG CSV, from Health > Export All Health Data > electrocardiograms/*.csv
    (header lines such as "Sample Rate,512 hertz", then one microvolt value per line). The header holds
    personal data (name, date of birth); only the sample rate, lead and unit are read, nothing else.
  * Polar Sensor Logger ECG export (H10 chest strap, 130 Hz): semicolon-separated, with a column
    whose name contains "ecg" and a timestamp column in ns or ms.
  * Generic CSV / TXT: either one numeric column (pass fs, e.g. an AD8232 + Arduino logger) or
    time + value columns (fs inferred from the timestamps; seconds, ms, us or ns are detected).

Layout: one folder per condition under data/sensing_gen/wearable/, e.g.
    still/  walking/  arm_motion/  car_passenger/
Every file in a folder is one recording of that condition. A file `fs.txt` in a folder overrides the
sample rate for headerless single-column files.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EXTS = {".csv", ".txt", ".tsv"}


@dataclass
class Recording:
    condition: str
    name: str            # file stem (no personal data)
    x: np.ndarray        # signal in millivolts when the unit is known, else raw units
    fs: float
    fmt: str
    lead: str = ""


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _apple_watch(lines: list[str]) -> tuple[np.ndarray, float, str] | None:
    head = "\n".join(lines[:15]).lower()
    if "sample rate" not in head:
        return None
    fs, lead, unit = None, "", "uv"
    vals = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        key = parts[0].lower()
        if key.startswith("sample rate") and len(parts) > 1:
            m = re.search(r"([\d.]+)", parts[1])
            fs = float(m.group(1)) if m else None
        elif key == "lead" and len(parts) > 1:
            lead = parts[1]
        elif key == "unit" and len(parts) > 1:
            unit = parts[1].lower()
        elif parts and _is_number(parts[0]):
            vals.append(float(parts[0]))
    if fs is None or len(vals) < 100:
        return None
    x = np.asarray(vals)
    if "µv" in unit or "uv" in unit or "microvolt" in unit:
        x = x / 1000.0
    return x, fs, lead


def _table(path: Path, fs_hint: float | None) -> tuple[np.ndarray, float, str]:
    txt = path.read_text(errors="ignore").splitlines()
    txt = [t for t in txt if t.strip()]
    sep = ";" if txt[0].count(";") > 0 else ("\t" if txt[0].count("\t") > 0 else ",")
    first = [c.strip() for c in txt[0].split(sep)]
    header = None if all(_is_number(c) for c in first if c) else [c.lower() for c in first]
    rows = [[c.strip() for c in r.split(sep)] for r in (txt[1:] if header else txt)]
    ncol = max(len(r) for r in rows)
    rows = [r for r in rows if len(r) == ncol]
    keep = [j for j in range(ncol) if all(_is_number(r[j]) for r in rows[: min(len(rows), 200)])]
    A = np.array([[float(r[j]) for j in keep] for r in rows if all(_is_number(r[j]) for j in keep)])
    if header:
        header = [header[j] if j < len(header) else "" for j in keep]
    if A.ndim == 1 or A.shape[1] == 1:
        if fs_hint is None:
            raise ValueError(f"{path.name}: single-column file needs a sample rate (put it in fs.txt)")
        return A.reshape(-1), float(fs_hint), "generic-1col"
    # choose the value column: named 'ecg' if present, else the last column
    vi = A.shape[1] - 1
    if header:
        for i, h in enumerate(header):
            if "ecg" in h:
                vi = i
    # timestamp column: first strictly increasing column that is not the value column
    fs = fs_hint
    unit_mv = 1.0
    if header and any("uv" in h or "µv" in h for h in header):
        unit_mv = 1e-3
    for i in range(A.shape[1]):
        if i == vi:
            continue
        d = np.diff(A[:, i])
        if (d > 0).mean() > 0.95:
            dt = float(np.median(d[d > 0]))
            h = header[i] if header else ""
            if "[ns]" in h or "ns" == h or dt > 1e5:
                dt *= 1e-9
            elif "[us]" in h or dt > 1e2:
                dt *= 1e-6
            elif "[ms]" in h or dt > 0.5:
                dt *= 1e-3
            fs = fs or 1.0 / dt
            break
    if fs is None:
        raise ValueError(f"{path.name}: could not infer the sample rate (put it in fs.txt)")
    fmt = "polar" if header and any("ecg" in h for h in header) else "generic-table"
    return A[:, vi] * unit_mv, float(fs), fmt


def load_file(path: Path, condition: str, fs_hint: float | None = None) -> Recording:
    lines = path.read_text(errors="ignore").splitlines()
    aw = _apple_watch(lines)
    if aw is not None:
        x, fs, lead = aw
        return Recording(condition, path.stem, x, fs, "apple-watch", lead)
    x, fs, fmt = _table(path, fs_hint)
    return Recording(condition, path.stem, x, fs, fmt)


def load_folder(root: str | Path) -> list[Recording]:
    root = Path(root)
    out = []
    for cond in sorted(p for p in root.iterdir() if p.is_dir()):
        fs_hint = None
        if (cond / "fs.txt").exists():
            fs_hint = float((cond / "fs.txt").read_text().strip())
        for f in sorted(cond.iterdir()):
            if f.suffix.lower() in EXTS and f.name != "fs.txt":
                try:
                    out.append(load_file(f, cond.name, fs_hint))
                except Exception as e:  # keep going, report
                    print(f"[wearable] skip {cond.name}/{f.name}: {e}")
    return out


def recorded_time(path: Path):
    """Apple Watch 'Recorded Date' (local wall-clock time, timezone suffix ignored) or None."""
    from datetime import datetime
    for ln in path.read_text(errors="ignore").splitlines()[:15]:
        if ln.lower().startswith("recorded date"):
            v = ln.split(",", 1)[1].strip()
            return datetime.strptime(v[:19], "%Y-%m-%d %H:%M:%S")
    return None


def sort_export(export_dir: str | Path, plan_csv: str | Path, out_root: str | Path) -> dict:
    """Copy Apple Watch ECG CSVs into condition folders using a plan of time blocks.

    plan.csv rows: condition,start,end  (local time 'YYYY-MM-DD HH:MM'), e.g.
        still,2026-11-05 09:00,2026-11-05 09:15
        walking,2026-11-05 09:20,2026-11-05 09:40
    Recordings outside every block are left out. Only the numeric part of each file is needed later,
    but files are copied unchanged; keep out_root inside the git-ignored data/ folder.
    """
    import shutil
    from datetime import datetime
    export_dir, out_root = Path(export_dir), Path(out_root)
    blocks = []
    for ln in Path(plan_csv).read_text().splitlines():
        p = [s.strip() for s in ln.split(",")]
        if len(p) == 3 and not p[0].lower().startswith("condition"):
            blocks.append((p[0], datetime.strptime(p[1], "%Y-%m-%d %H:%M"), datetime.strptime(p[2], "%Y-%m-%d %H:%M")))
    counts = {}
    for f in sorted(export_dir.glob("*.csv")):
        t = recorded_time(f)
        for cond, a, b in blocks:
            if t is not None and a <= t <= b:
                (out_root / cond).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out_root / cond / f.name)
                counts[cond] = counts.get(cond, 0) + 1
                break
    return counts


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Sort an Apple Watch ECG export into condition folders.")
    ap.add_argument("export_dir", help="apple_health_export/electrocardiograms")
    ap.add_argument("plan", help="CSV of condition,start,end")
    ap.add_argument("--out", default="data/sensing_gen/wearable")
    a = ap.parse_args()
    print(sort_export(a.export_dir, a.plan, a.out))
