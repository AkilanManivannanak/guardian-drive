"""Step 8 (CPU, ~10 s): cross-check the finite-volume electrode solver with an independent P1
finite-element solver on the COMSOL-equivalent configuration (see multiphysics/comsol/README.md)."""
from __future__ import annotations

import numpy as np

from ..multiphysics.compare_comsol import python_reference
from ..multiphysics.electrode_skin_fem import fem_impedance
from .common import env_info, get_profile, out_dir, save_json


def main():
    od = out_dir(get_profile())
    f = np.logspace(0, 4, 21)
    z_fv = python_reference(f)
    z_fe = np.array([fem_impedance(x) for x in f])
    rel = np.abs(np.abs(z_fe) / np.abs(z_fv) - 1)
    dph = np.abs(np.angle(z_fe, deg=True) - np.angle(z_fv, deg=True))
    conv = [{"h_mm": h * 1e3, "dz0_um": dz * 1e6, "Z_10Hz_ohm": float(abs(fem_impedance(10.0, h=h, dz0=dz)))}
            for h, dz in ((0.5e-3, 20e-6), (0.25e-3, 10e-6), (0.125e-3, 5e-6))]
    lift_fe = abs(fem_impedance(10.0, lift=0.2)) / abs(fem_impedance(10.0)) - 1
    lift_fv = abs(python_reference([10.0], lift=0.2)[0]) / abs(python_reference([10.0])[0]) - 1
    res = {"env": env_info(), "frequency_hz": f, "fv_abs_ohm": np.abs(z_fv), "fe_abs_ohm": np.abs(z_fe),
           "rel_error": rel, "max_rel_error": float(rel.max()), "mean_rel_error": float(rel.mean()),
           "max_phase_diff_deg": float(dph.max()), "fe_mesh_convergence": conv,
           "liftoff20_rise_fe": float(lift_fe), "liftoff20_rise_fv": float(lift_fv)}
    save_json(res, od / "fem_crosscheck.json")
    print(f"[fem] FE vs FV |Z|: max {100 * rel.max():.2f} %, mean {100 * rel.mean():.2f} %, phase within "
          f"{dph.max():.2f} deg; 20 % lift-off: FE {100 * lift_fe:.2f} % vs FV {100 * lift_fv:.2f} %", flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.loglog(f, np.abs(z_fv), color="#2a78d6", lw=2, label="finite volume (main solver)")
        ax.loglog(f, np.abs(z_fe), "o", color="#eb6834", ms=5, label="finite element (independent check)")
        ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("|Z| two-electrode [Ω]")
        ax.set_title(f"Wet electrode on layered tissue: FE vs FV, max |Z| difference {100 * rel.max():.2f} %", fontsize=10)
        ax.grid(alpha=0.3, which="both"); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(od / "fig_fem_crosscheck.png", dpi=130)
    except Exception as e:
        print("[fem] figure skipped:", e)


if __name__ == "__main__":
    main()
