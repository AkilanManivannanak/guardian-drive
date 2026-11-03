"""Cross-check the finite-volume electrode model against a COMSOL Multiphysics export.

    python -m sensing_gen.multiphysics.compare_comsol --csv comsol_wet_sweep.csv \
        [--liftoff-csv comsol_wet_liftoff8mm.csv] [--update-readme]

The CSV is COMSOL's table export (lines starting with '%' are ignored), with columns
    frequency [Hz], |Z| [ohm] (or terminal current magnitude with --current), optional phase [deg].
The Python model is evaluated in the exact configuration of comsol/README.md: wet electrode,
gel-hydrated stratum corneum as a contact impedance, no electrode double layer, no gel layer, no
pressure.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .electrode_skin_fv import ElectrodeConfig, TissueModel, make_grid, solve_impedance

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "sensing_gen"
MARK_BEGIN, MARK_END = "<!-- COMSOL-CHECK:BEGIN -->", "<!-- COMSOL-CHECK:END -->"


def comsol_equivalent() -> tuple[TissueModel, ElectrodeConfig]:
    return TissueModel(), ElectrodeConfig(kind="wet", interface_r=1e-12, interface_c=0.0, gel_thickness=1e-9)


def python_reference(freqs, lift: float = 0.0) -> np.ndarray:
    tm, ec = comsol_equivalent()
    g = make_grid(tm)
    return np.array([solve_impedance(tm, ec, f, pressure=(0.0, 0.0), lift_off=(lift, 0.0), grid=g) for f in freqs])


def read_table(path: Path, current: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rows = []
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("%") or ln.startswith("#"):
            continue
        vals = [v for v in re.split(r"[,\s;]+", ln) if v]
        try:
            rows.append([float(v.replace("i", "j")) if "i" not in v else abs(complex(v.replace("i", "j"))) for v in vals])
        except ValueError:
            continue  # header line
    a = np.array(rows, dtype=float)
    f, mag = a[:, 0], np.abs(a[:, 1])
    if current:
        mag = 1.0 / mag  # 1 V terminal: |Z| = 1 / |I|
    ph = a[:, 2] if a.shape[1] > 2 else None
    return f, mag, ph


def update_readme(path: Path, text: str) -> None:
    s = path.read_text()
    if MARK_BEGIN in s and MARK_END in s:
        pre, rest = s.split(MARK_BEGIN, 1)
        _, post = rest.split(MARK_END, 1)
        path.write_text(pre + MARK_BEGIN + "\n" + text + "\n" + MARK_END + post)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--liftoff-csv")
    ap.add_argument("--current", action="store_true", help="column 2 is terminal current, not impedance")
    ap.add_argument("--update-readme", action="store_true")
    ap.add_argument("--comsol-version", default="")
    a = ap.parse_args(argv)

    f, z_c, ph_c = read_table(Path(a.csv), a.current)
    z_p = python_reference(f)
    rel = np.abs(z_c / np.abs(z_p) - 1)
    res = {"frequency_hz": f, "comsol_abs_ohm": z_c, "python_abs_ohm": np.abs(z_p),
           "python_phase_deg": np.angle(z_p, deg=True), "rel_error": rel,
           "max_rel_error": float(rel.max()), "mean_rel_error": float(rel.mean()),
           "comsol_version": a.comsol_version}
    if ph_c is not None:
        res["comsol_phase_deg"] = ph_c
        res["max_phase_diff_deg"] = float(np.max(np.abs(ph_c - np.angle(z_p, deg=True))))
    if a.liftoff_csv:
        fl, zl_c, _ = read_table(Path(a.liftoff_csv), a.current)
        i10c = int(np.argmin(abs(f - 10))); j10c = int(np.argmin(abs(fl - 10)))
        rise_c = zl_c[j10c] / z_c[i10c] - 1
        rise_p = abs(python_reference([10.0], lift=0.2)[0]) / abs(python_reference([10.0])[0]) - 1
        res["liftoff_rise_comsol"] = float(rise_c)
        res["liftoff_rise_python"] = float(rise_p)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comsol_comparison.json").write_text(json.dumps(
        {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()}, indent=2))

    line = (f"**COMSOL cross-check:** a COMSOL Multiphysics{(' ' + a.comsol_version) if a.comsol_version else ''} "
            f"2-D model (AC/DC, Electric Currents, frequency domain, contact impedance) agrees with the Python "
            f"finite-volume model to **{100 * res['max_rel_error']:.1f} % max / {100 * res['mean_rel_error']:.1f} % mean** "
            f"|Z| error over {f.min():g}–{f.max():g} Hz")
    if "liftoff_rise_comsol" in res:
        line += (f"; 20 % electrode lift-off raises |Z| by {100 * res['liftoff_rise_comsol']:.1f} % in COMSOL vs "
                 f"{100 * res['liftoff_rise_python']:.1f} % in Python")
    line += " ([`comsol_comparison.json`](outputs/sensing_gen/comsol_comparison.json))."
    print(line)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.loglog(f, np.abs(z_p), color="#1971c2", lw=2, label="Python finite volume")
        ax.loglog(f, z_c, "o", color="#d6336c", ms=5, label="COMSOL")
        ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("|Z| two-electrode [Ω]")
        ax.set_title(f"Wet electrode on layered tissue: max error {100 * res['max_rel_error']:.1f} %")
        ax.grid(alpha=0.3, which="both"); ax.legend()
        fig.tight_layout(); fig.savefig(OUT / "fig_comsol.png", dpi=130)
    except Exception as e:
        print("figure skipped:", e)
    if a.update_readme:
        update_readme(ROOT / "sensing_gen" / "README.md", line.replace("(outputs/", "(../outputs/"))
        update_readme(ROOT / "README.md", line)
        print("README files updated between the COMSOL-CHECK markers.")


if __name__ == "__main__":
    main()
