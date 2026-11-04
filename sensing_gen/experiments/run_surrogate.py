"""Step 11: neural surrogate of the electrode-skin simulator with calibrated uncertainty.

Dataset: Latin-hypercube samples of 10 tissue/contact/geometry parameters (space.py), each simulated by
the finite-element solver over 21 frequencies (1 Hz-10 kHz). Models (5-member deep ensembles):
  * plain   - MLP predicts log10|Z| and phase directly
  * hybrid  - physics-informed: MLP predicts the correction to an analytic contact-impedance prior
Baselines: quadratic ridge regression (plain and hybrid). Reported on a held-out test set: |Z| and phase
error, 95 % interval coverage, data efficiency (250 ... N training simulations), an out-of-range test
(every parameter range widened by 25 %) to check that the ensemble's uncertainty flags extrapolation,
the FE discretisation floor (coarse vs fine mesh) and the speed-up over the solver.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..multiphysics.electrode_skin_fem import fem_spectrum
from ..surrogate.dataset import build, from_targets, targets
from ..surrogate.model import save_surrogate, train_surrogate
from ..surrogate.space import FREQS, contact_prior, to_configs, to_unit
from .common import DATA, env_info, get_profile, out_dir, save_json


def errors(pred_m, pred_s, y):
    n = y.shape[1] // 2
    rel = np.abs(10 ** (pred_m[:, :n] - y[:, :n]) - 1)
    ph = np.degrees(np.abs(pred_m[:, n:] - y[:, n:]))
    out = {"absZ_rel_err_median": float(np.median(rel)), "absZ_rel_err_p95": float(np.percentile(rel, 95)),
           "absZ_rel_err_max": float(rel.max()), "phase_err_deg_median": float(np.median(ph)),
           "phase_err_deg_p95": float(np.percentile(ph, 95))}
    if pred_s is not None:
        z = np.abs(pred_m - y) / np.maximum(pred_s, 1e-12)
        out["coverage_95"] = float((z <= 1.96).mean())
        per = np.abs(pred_m - y).mean(1)
        from scipy.stats import spearmanr
        out["spearman_sigma_vs_error"] = float(spearmanr(pred_s.mean(1), per).correlation)
        out["mean_sigma_log10Z"] = float(pred_s[:, :n].mean())
    return out


def ridge_quad(theta_tr, y_tr, theta_te, hybrid):
    def feats(th):
        x = 2 * to_unit(th) - 1
        i, j = np.triu_indices(x.shape[1])
        return np.c_[np.ones(len(x)), x, x[:, i] * x[:, j]]
    off_tr = contact_prior(theta_tr) if hybrid else 0
    off_te = contact_prior(theta_te) if hybrid else 0
    F = feats(theta_tr)
    W = np.linalg.solve(F.T @ F + 1e-3 * np.eye(F.shape[1]), F.T @ (y_tr - off_tr))
    return feats(theta_te) @ W + off_te


def main():
    prof = get_profile()
    od = out_dir(prof)
    full = prof.name == "full"
    n_tr, n_te, n_ood = (4000, 1000, 300) if full else (300, 100, 50)
    steps = 6000 if full else 400
    curve = (250, 500, 1000, 2000, 4000) if full else (100, 300)
    cache = DATA / "surrogate"
    torch.set_num_threads(max(1, torch.get_num_threads()))

    t0 = time.time()
    tr = build(cache / f"train_{n_tr}.npz", n_tr, seed=0)
    te = build(cache / f"test_{n_te}.npz", n_te, seed=1)
    ood = build(cache / f"ood_{n_ood}.npz", n_ood, seed=2, expand=0.25)
    sim_wall = float(tr["sim_seconds_total"])
    print(f"[surrogate] datasets ready ({time.time() - t0:.0f} s); FE wall time for {n_tr} spectra "
          f"{sim_wall:.0f} s", flush=True)
    ytr, yte, yood = targets(tr["Z"]), targets(te["Z"]), targets(ood["Z"])
    res = {"env": env_info(), "n_train": n_tr, "n_test": n_te, "n_ood": n_ood, "frequencies_hz": FREQS,
           "models": {}, "data_efficiency": {}}

    # FE discretisation floor: coarse (dataset) vs fine mesh on a few test samples
    k = 12 if full else 3
    fine = np.array([fem_spectrum(FREQS, *to_configs(t)[:2], lift=to_configs(t)[2], h=0.25e-3, dz0=10e-6,
                                  ratio=1.12) for t in te["theta"][:k]])
    fl = np.abs(np.abs(fine) / np.abs(te["Z"][:k]) - 1)
    res["fe_mesh_floor"] = {"n": k, "absZ_rel_diff_median": float(np.median(fl)), "absZ_rel_diff_max": float(fl.max())}
    print(f"[surrogate] FE coarse-vs-fine mesh |Z| difference: median {np.median(fl):.2e}, max {fl.max():.2e}", flush=True)

    # baselines
    for hyb in (False, True):
        name = f"quadratic_ridge_{'hybrid' if hyb else 'plain'}"
        res["models"][name] = errors(ridge_quad(tr["theta"], ytr, te["theta"], hyb), None, yte)
    res["models"]["analytic_contact_prior_only"] = errors(contact_prior(te["theta"]), None, yte)

    # deep ensembles
    models = {}
    for hyb in (False, True):
        name = f"ensemble_{'hybrid' if hyb else 'plain'}"
        t1 = time.time()
        m = train_surrogate(tr["theta"], ytr, k=5, steps=steps, hybrid=hyb, seed=0)
        models[name] = m
        pm, ps = m.predict(te["theta"])
        r = errors(pm, ps, yte)
        r["train_seconds"] = round(time.time() - t1, 1)
        pm_o, ps_o = m.predict(ood["theta"])
        r["ood"] = errors(pm_o, ps_o, yood)
        res["models"][name] = r
        print(f"[surrogate] {name:<16} |Z| err median {100 * r['absZ_rel_err_median']:.2f} % p95 "
              f"{100 * r['absZ_rel_err_p95']:.2f} % | phase median {r['phase_err_deg_median']:.3f} deg | "
              f"95% coverage {r['coverage_95']:.2f} | OOD |Z| median {100 * r['ood']['absZ_rel_err_median']:.2f} % "
              f"with sigma x{r['ood']['mean_sigma_log10Z'] / r['mean_sigma_log10Z']:.1f}", flush=True)
        save_json(res, od / "surrogate_results.json")
    for name, r in res["models"].items():
        if name.startswith("quad") or name.startswith("analytic"):
            print(f"[surrogate] {name:<28} |Z| err median {100 * r['absZ_rel_err_median']:.2f} % p95 "
                  f"{100 * r['absZ_rel_err_p95']:.2f} %", flush=True)
    save_surrogate(models["ensemble_hybrid"], od / "ckpt_surrogate.pt")

    # speed
    big = np.repeat(te["theta"], max(1, 10000 // n_te), 0)
    m = models["ensemble_hybrid"]
    m.predict(big[:100])
    t1 = time.time(); m.predict(big); sur_t = (time.time() - t1) / len(big)
    t1 = time.time(); [fem_spectrum(FREQS, *to_configs(t)[:2], lift=to_configs(t)[2]) for t in te["theta"][:20]]
    fe_t = (time.time() - t1) / 20
    res["speed"] = {"fe_seconds_per_spectrum_single_core": fe_t, "surrogate_seconds_per_spectrum_batched": sur_t,
                    "speedup": fe_t / sur_t}
    print(f"[surrogate] FE {fe_t * 1e3:.0f} ms/spectrum vs surrogate {sur_t * 1e6:.1f} us/spectrum "
          f"-> {fe_t / sur_t:,.0f}x faster", flush=True)

    # data efficiency
    for n in curve:
        if n > n_tr:
            continue
        res["data_efficiency"][str(n)] = {}
        for hyb in (False, True):
            mm = train_surrogate(tr["theta"][:n], ytr[:n], k=3, steps=steps, hybrid=hyb, seed=1)
            pm, ps = mm.predict(te["theta"])
            e = errors(pm, ps, yte)
            res["data_efficiency"][str(n)]["hybrid" if hyb else "plain"] = e
        d = res["data_efficiency"][str(n)]
        print(f"[surrogate] n={n:<5} plain {100 * d['plain']['absZ_rel_err_median']:.2f} % | hybrid "
              f"{100 * d['hybrid']['absZ_rel_err_median']:.2f} %", flush=True)
        save_json(res, od / "surrogate_results.json")
    save_json(res, od / "surrogate_results.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ns = [int(n) for n in res["data_efficiency"]]
        for key, col in (("plain", "#888"), ("hybrid", "#2a78d6")):
            ax[0].loglog(ns, [100 * res["data_efficiency"][str(n)][key]["absZ_rel_err_median"] for n in ns],
                         "o-", color=col, label=f"{key} ensemble")
        ax[0].set_xlabel("FE training simulations"); ax[0].set_ylabel("median |Z| error [%]")
        ax[0].set_title("Data efficiency: physics prior + learned correction", fontsize=10)
        ax[0].grid(alpha=0.3, which="both"); ax[0].legend(frameon=False)
        pm, ps = models["ensemble_hybrid"].predict(te["theta"][:3])
        Zp = from_targets(pm)
        for i, c in enumerate(("#2a78d6", "#eb6834", "#2b9e5e")):
            ax[1].loglog(FREQS, np.abs(te["Z"][i]), "o", color=c, ms=4)
            ax[1].loglog(FREQS, np.abs(Zp[i]), "-", color=c, lw=1.5)
            lo, hi = 10 ** (pm[i, :21] - 1.96 * ps[i, :21]), 10 ** (pm[i, :21] + 1.96 * ps[i, :21])
            ax[1].fill_between(FREQS, lo, hi, color=c, alpha=0.2)
        ax[1].set_xlabel("frequency [Hz]"); ax[1].set_ylabel("|Z| [Ω]")
        ax[1].set_title("Held-out tissues: FE (dots) vs surrogate ± 95 %", fontsize=10); ax[1].grid(alpha=0.3, which="both")
        pm, ps = models["ensemble_hybrid"].predict(np.concatenate([te["theta"], ood["theta"]]))
        yy = np.concatenate([yte, yood])
        err = np.abs(10 ** (pm[:, :21] - yy[:, :21]) - 1).mean(1) * 100
        sig = ps[:, :21].mean(1)
        ax[2].loglog(sig[:n_te], err[:n_te], ".", color="#2a78d6", alpha=0.5, label="in range")
        ax[2].loglog(sig[n_te:], err[n_te:], ".", color="#d6336c", alpha=0.7, label="ranges +25 % (extrapolation)")
        ax[2].set_xlabel("ensemble σ (log10|Z|)"); ax[2].set_ylabel("actual |Z| error [%]")
        ax[2].set_title("Uncertainty tracks error and flags extrapolation", fontsize=10)
        ax[2].grid(alpha=0.3, which="both"); ax[2].legend(frameon=False)
        fig.tight_layout(); fig.savefig(od / "fig_surrogate.png", dpi=130)
    except Exception as e:
        print("[surrogate] figure skipped:", e)


if __name__ == "__main__":
    main()
