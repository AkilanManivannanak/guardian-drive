#!/usr/bin/env python3
"""Render a short, clearly labeled replay from Guardian Drive's synthetic sensor simulator.

Requires NumPy (already used by the simulator), R with base PNG graphics, and FFmpeg.
The rendered replay contains no real participant data and no ML predictions.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acquisition.models import FS_ECG, FS_EDA, FS_IMU, FS_RESP
from acquisition.simulator import GuardianSimulator


DEFAULT_OUTPUT = ROOT / "docs" / "media"
DURATION = 10.0
POINTS = 500
FRAMES = 48
FPS = 12


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (2.0 * (values - low) / (high - low)) - 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenario", default="drowsy", choices=("normal", "drowsy", "crash_severe"))
    args = parser.parse_args()

    rscript, ffmpeg = shutil.which("Rscript"), shutil.which("ffmpeg")
    if not rscript or not ffmpeg:
        parser.error("Rscript and ffmpeg must be on PATH")

    frame = GuardianSimulator(
        args.scenario,
        duration=DURATION,
        subject_id="readme-demo",
        session_id=f"readme-demo-{args.scenario}-v1",
    ).window(0.0, DURATION)

    source_time = {
        "ecg": np.arange(len(frame.ecg), dtype=np.float64) / FS_ECG,
        "eda": np.arange(len(frame.eda), dtype=np.float64) / FS_EDA,
        "respiration": np.arange(len(frame.respiration), dtype=np.float64) / FS_RESP,
        "accel_x": np.arange(len(frame.accel), dtype=np.float64) / FS_IMU,
    }
    signals = {
        "ecg": frame.ecg,
        "eda": frame.eda,
        "respiration": frame.respiration,
        "accel_x": frame.accel[:, 0],
    }
    sample_time = np.linspace(0.0, DURATION, POINTS)
    resampled = {
        name: _normalise(np.interp(sample_time, source_time[name], signal))
        for name, signal in signals.items()
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="guardian-readme-demo-") as temp:
        temp_dir = Path(temp)
        csv_path = temp_dir / "synthetic_signals.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("time", *signals.keys()))
            for index, t in enumerate(sample_time):
                row = [f"{t:.5f}"]
                for name in signals:
                    row.append(f"{resampled[name][index]:.6f}")
                writer.writerow(row)

        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()
        subprocess.run(
            [rscript, str(ROOT / "scripts" / "render_readme_demo.R"), str(csv_path), str(frames_dir), str(FRAMES), args.scenario],
            check=True,
        )
        mp4_path = output_dir / "guardian-synthetic-replay.mp4"
        gif_path = output_dir / "guardian-synthetic-replay.gif"
        subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(FPS),
             "-i", str(frames_dir / "frame-%03d.png"), "-c:v", "libx264", "-crf", "23",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4_path)],
            check=True,
        )
        subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(FPS),
             "-i", str(frames_dir / "frame-%03d.png"), "-filter_complex",
             "fps=12,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
             "-loop", "0", str(gif_path)],
            check=True,
        )
    print(f"Wrote {gif_path.relative_to(ROOT)} and {mp4_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
