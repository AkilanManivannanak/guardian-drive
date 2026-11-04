"""Step 12: robust electrode design through the differentiable surrogate, verified by the FE solver.

Maximise J(w, s) = E[ln s - 0.5 ln Re Z(10 Hz)] (an SNR proxy, see surrogate/design.py) over electrode
width w and spacing s under a 35 mm patch-length budget and a 5 mm gap, averaged over a population of
tissues and lift-off states. Compared with (a) brute-force FE grid search over the same feasible set,
(b) the design optimised for one nominal tissue, (c) the 'maximise spacing' heuristic (w = 5, s = 30 mm)
and (d) a common layout (w = 10, s = 25 mm), which turns out to be close to optimal.
"""
from __future__ import annotations

import time

import numpy as np

from ..surrogate.design import fe_objective, optimise, population, project
from ..surrogate.model import load_surrogate
from .common import env_info, get_profile, out_dir, save_json

L, GAP = 35.0, 5.0


def main():
    prof = get_profile()
    od = out_dir(prof)
    full = prof.name == "full"
    model = load_surrogate(od / "ckpt_surrogate.pt")
    pop_opt = population(512 if full else 64, seed=123)
    pop_fe = population(48 if full else 6, seed=456)          # independent population for FE checks
    nom = population(8, nominal=True)

    t0 = time.time()
    rob = optimise(model, pop_opt, L, GAP)
    t_sur = time.time() - t0
    nomd = optimise(model, nom, L, GAP)
    print(f"[design] surrogate optimum (robust): w={rob['w_mm']:.2f} mm s={rob['s_mm']:.2f} mm "
          f"({t_sur:.1f} s); nominal-tissue optimum: w={nomd['w_mm']:.2f} s={nomd['s_mm']:.2f}", flush=True)

    # brute-force FE grid on the feasible set
    t0 = time.time()
    step = 1.0 if full else 2.5
    grid = []
    for w in np.arange(5.0, 15.0 + 1e-9, step):
        for s in np.arange(20.0, 40.0 + 1e-9, step):
            if s + w <= L + 1e-9 and s - w >= GAP - 1e-9:
                grid.append((w, s, float(fe_objective(pop_fe, w, s).mean())))
    t_grid = time.time() - t0
    gbest = max(grid, key=lambda g: g[2])
    print(f"[design] FE grid ({len(grid)} designs x {len(pop_fe)} tissues, {t_grid:.0f} s): best w={gbest[0]:.1f} "
          f"s={gbest[1]:.1f} J={gbest[2]:.4f}", flush=True)

    designs = {"surrogate_robust": (rob["w_mm"], rob["s_mm"]), "surrogate_nominal_tissue": (nomd["w_mm"], nomd["s_mm"]),
               "fe_grid_best": (gbest[0], gbest[1]), "common_w10_s25": (10.0, 25.0),
               "max_spacing_w5_s30": (5.0, 30.0)}
    ev = {}
    for k, (w, s) in designs.items():
        j = fe_objective(pop_fe, w, s)
        ev[k] = {"w_mm": w, "s_mm": s, "J_fe_mean": float(j.mean()), "J_fe_p10": float(np.percentile(j, 10))}
    db = 20 / np.log(10)
    for k in ev:
        ev[k]["snr_gain_db_vs_max_spacing"] = float((ev[k]["J_fe_mean"] - ev["max_spacing_w5_s30"]["J_fe_mean"]) * db)
        ev[k]["snr_gain_db_vs_common"] = float((ev[k]["J_fe_mean"] - ev["common_w10_s25"]["J_fe_mean"]) * db)
        ev[k]["regret_vs_fe_grid"] = float(ev["fe_grid_best"]["J_fe_mean"] - ev[k]["J_fe_mean"])
        print(f"[design] {k:<26} w={ev[k]['w_mm']:5.2f} s={ev[k]['s_mm']:5.2f} | FE J mean {ev[k]['J_fe_mean']:.4f} "
              f"p10 {ev[k]['J_fe_p10']:.4f} | SNR vs max-spacing {ev[k]['snr_gain_db_vs_max_spacing']:+.2f} dB, vs 10/25 "
              f"{ev[k]['snr_gain_db_vs_common']:+.2f} dB | "
              f"regret {ev[k]['regret_vs_fe_grid']:.4f}", flush=True)
    res = {"env": env_info(), "budget_mm": L, "gap_mm": GAP, "n_pop_opt": len(pop_opt), "n_pop_fe": len(pop_fe),
           "surrogate_opt_seconds": t_sur, "fe_grid_seconds": t_grid, "fe_grid_n_designs": len(grid),
           "fe_grid": [{"w_mm": g[0], "s_mm": g[1], "J": g[2]} for g in grid], "designs": ev,
           "J_surrogate_at_optimum": rob["J_surrogate"]}
    save_json(res, od / "design_results.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        g = np.array(grid)
        sc = ax.scatter(g[:, 0], g[:, 1], c=g[:, 2], cmap="viridis", s=60, marker="s")
        plt.colorbar(sc, label="FE objective J (higher = better SNR)")
        for tr in rob["traces"]:
            t = np.array(tr)
            ax.plot(t[:, 0], t[:, 1], "-", color="white", lw=1)
            ax.plot(t[0, 0], t[0, 1], "o", color="white", ms=4)
        ax.plot(rob["w_mm"], rob["s_mm"], "*", color="#d6336c", ms=16, label="surrogate gradient optimum")
        ax.plot(gbest[0], gbest[1], "x", color="black", ms=10, mew=2, label="FE grid best")
        ax.plot(10, 25, "D", color="#eb6834", ms=7, label="common 10/25 mm")
        ax.plot(5, 30, "s", color="#888", ms=7, label="max-spacing 5/30 mm")
        ax.set_xlabel("electrode width w [mm]"); ax.set_ylabel("spacing s [mm]")
        ax.set_title("Robust electrode design via differentiable surrogate", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        fig.tight_layout(); fig.savefig(od / "fig_design.png", dpi=130)
    except Exception as e:
        print("[design] figure skipped:", e)


if __name__ == "__main__":
    main()
