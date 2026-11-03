"""Step 0 (CPU only, ~30 s): electro-mechanical multiphysics model of the electrode-skin interface.

Produces impedance spectra for wet vs dry electrodes, motion sensitivities (kappa) and mains
conversion, verifies the solver (analytic thin-strip case, grid convergence, current conservation),
and checks the prior ranges of the lumped sensor model in sensing_gen/physics/sensor.py.
"""
from __future__ import annotations

import time

import numpy as np

from ..multiphysics.electrode_skin_fv import (ElectrodeConfig, Layer, TissueModel, contact_state,
                                              impedance_spectrum, interface_admittance, make_grid,
                                              motion_sensitivity, solve_impedance)
from .common import env_info, get_profile, out_dir, save_json


def verification() -> dict:
    out = {}
    # 1) analytic thin-strip limit: uniform injection under each electrode (transmission-line model)
    tm = TissueModel(layers=[Layer("strip", 2e-4, 0.2, 1.0)])
    ec = ElectrodeConfig(kind="dry")
    f = 0.01
    Z = solve_impedance(tm, ec, f)
    Y = interface_admittance(ec, tm, contact_state(ec, tm, 5e3), 2 * np.pi * f)
    st = 0.2 * 2e-4 * ec.length
    Zan = 2 / (Y * ec.width * ec.length) + (ec.spacing - ec.width) / st + 2 * ec.width / (3 * st)
    out["thin_strip_rel_error"] = float(abs(abs(Z) / abs(Zan) - 1))
    # 2) grid convergence on the full layered model (dry electrode, 10 Hz)
    tm, ec = TissueModel(), ElectrodeConfig(kind="dry")
    conv = []
    for nx, dz0 in ((120, 20e-6), (240, 10e-6), (480, 5e-6)):
        g = make_grid(tm, nx=nx, dz0=dz0)
        conv.append({"nx": nx, "nz": len(g[1]) - 1, "dz0_um": dz0 * 1e6,
                     "Z_ohm": float(abs(solve_impedance(tm, ec, 10.0, grid=g)))})
    ref = conv[-1]["Z_ohm"]
    for c in conv:
        c["rel_diff_vs_finest"] = abs(c["Z_ohm"] / ref - 1)
    out["grid_convergence"] = conv
    # 3) reciprocity: swapping the electrodes leaves Z unchanged
    za = solve_impedance(tm, ec, 10.0, pressure=(3e3, 8e3))
    zb = solve_impedance(tm, ec, 10.0, pressure=(8e3, 3e3))
    out["reciprocity_rel_error"] = float(abs(za - zb) / abs(za))
    return out


def main():
    prof = get_profile()
    od = out_dir(prof)
    t0 = time.time()
    tm = TissueModel()
    freqs = np.logspace(0, 4, 21)
    res = {"env": env_info(), "verification": verification(), "electrodes": {}}
    p_grid = np.array([1e3, 2e3, 3e3, 5e3, 8e3, 12e3, 20e3, 30e3])
    lift_grid = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8])
    grid = make_grid(tm)
    for kind in ("wet", "dry"):
        ec = ElectrodeConfig(kind=kind)
        Zs = impedance_spectrum(tm, ec, freqs)
        sens = motion_sensitivity(tm, ec)
        z_p = [abs(solve_impedance(tm, ec, 10.0, pressure=(p, 5e3), grid=grid)) for p in p_grid]
        z_l = [abs(solve_impedance(tm, ec, 10.0, lift_off=(l, 0.0), grid=grid)) for l in lift_grid]
        # mains: 60 Hz common-mode voltage converted to differential by the impedance mismatch
        z60_0 = solve_impedance(tm, ec, 60.0, grid=grid)
        z60_l = solve_impedance(tm, ec, 60.0, lift_off=(0.2, 0.0), grid=grid)
        mains = {f"zin_{int(zin / 1e6)}MOhm": float(abs(z60_l - z60_0) / 2 / zin) for zin in (10e6, 100e6)}
        res["electrodes"][kind] = {
            "spectrum_hz": freqs, "spectrum_abs_ohm": np.abs(Zs), "spectrum_phase_deg": np.angle(Zs, deg=True),
            "Z_10Hz_ohm": float(abs(Zs[np.argmin(abs(freqs - 10))])),
            "sensitivity": sens,
            "pressure_sweep_pa": p_grid, "Z_vs_pressure_ohm": z_p,
            "liftoff_sweep": lift_grid, "Z_vs_liftoff_ohm": z_l,
            "mains_diff_per_volt_cm_lift20pct": mains,
            "ze_over_zin": {f"zin_{int(zin / 1e6)}MOhm": float(abs(Zs[np.argmin(abs(freqs - 10))]) / 2 / zin)
                            for zin in (10e6, 100e6)},
        }
        print(f"[multiphysics] {kind}: |Z|(10 Hz) {res['electrodes'][kind]['Z_10Hz_ohm'] / 1e3:.1f} kOhm, "
              f"kappa_pressure {sens['kappa_pressure']:.3f}, kappa_liftoff {sens['kappa_liftoff']:.3f}", flush=True)

    # does the lumped sensor model's prior cover what the field model predicts?
    # priors of physics/sensor.py: kappa ~ U(0, 0.5), Z_e0/Z_in ~ U(0.001, 0.05) (kept in sync by a test)
    kappas = [res["electrodes"][k]["sensitivity"][m] for k in ("wet", "dry")
              for m in ("kappa_pressure", "kappa_liftoff")]
    ratios = [v for k in ("wet", "dry") for v in res["electrodes"][k]["ze_over_zin"].values()]
    res["sensor_prior_check"] = {
        "kappa_prior": [0.0, 0.5], "kappa_multiphysics_range": [min(kappas), max(kappas)],
        "kappa_covered": bool(max(kappas) <= 0.5),
        "ze_over_zin_prior": [0.001, 0.05], "ze_over_zin_multiphysics_range": [min(ratios), max(ratios)],
        "ze_over_zin_covered": bool(min(ratios) >= 0.001 and max(ratios) <= 0.05),
    }
    res["seconds"] = round(time.time() - t0, 1)
    print("[multiphysics] verification:", {k: v for k, v in res["verification"].items() if k != "grid_convergence"},
          flush=True)
    print("[multiphysics] sensor prior check:", res["sensor_prior_check"], flush=True)
    save_json(res, od / "multiphysics_results.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        # potential field (dry, 10 Hz), top 12 mm
        ec = ElectrodeConfig(kind="dry")
        _, phi, (xc, zc) = solve_impedance(tm, ec, 10.0, return_field=True)
        m = zc < 12e-3
        X, Zg = np.meshgrid(xc * 1e3, zc[m] * 1e3, indexing="ij")
        cs = ax[0].contourf(X, Zg, np.real(phi[:, m]), 30, cmap="RdBu_r")
        ax[0].invert_yaxis(); ax[0].set_xlabel("x [mm]"); ax[0].set_ylabel("depth [mm]")
        for d in np.cumsum([l.thickness for l in tm.layers])[:-1]:
            ax[0].axhline(d * 1e3, color="k", lw=0.6, ls="--")
        ax[0].set_title("EQS potential, dry electrodes, 10 Hz")
        fig.colorbar(cs, ax=ax[0], label="Re φ [V]")
        for kind, c in (("wet", "#1971c2"), ("dry", "#d6336c")):
            e = res["electrodes"][kind]
            ax[1].loglog(freqs, e["spectrum_abs_ohm"], color=c, label=kind)
            ax[2].plot(lift_grid * 100, np.array(e["Z_vs_liftoff_ohm"]) / e["Z_vs_liftoff_ohm"][0], color=c,
                       marker="o", label=f"{kind}: lift-off")
            ax[2].plot((1 - p_grid / 5e3) * 100, np.array(e["Z_vs_pressure_ohm"]) / e["Z_vs_pressure_ohm"][3],
                       color=c, ls="--", marker="s", label=f"{kind}: unloading")
        ax[1].set_xlabel("frequency [Hz]"); ax[1].set_ylabel("|Z| two-electrode [Ω]"); ax[1].legend()
        ax[1].set_title("Electrode-skin-tissue impedance"); ax[1].grid(alpha=0.3, which="both")
        ax[2].set_xlabel("perturbation of one electrode [%]"); ax[2].set_ylabel("Z / Z₀ (10 Hz)")
        ax[2].set_title("Motion sensitivity (contact mechanics → EQS)"); ax[2].legend(fontsize=8)
        ax[2].grid(alpha=0.3); ax[2].set_xlim(-10, 85)
        fig.tight_layout()
        fig.savefig(od / "fig_multiphysics.png", dpi=130)
    except Exception as e:
        print("[multiphysics] figure skipped:", e)


if __name__ == "__main__":
    main()
