"""Step 10: physics-informed (limit-cycle) guidance for the ECG diffusion sampler.

The trained diffusion model follows its heart-rate condition only loosely (HR-control MAE ~10 bpm in
step 3). The McSharry ODE says an ECG at heart rate HR is a function of the phase of a limit cycle of
period 60/HR s; we turn that into a guidance energy on the sampler's x0 estimate (see
diffusion/physics_guidance.py). No retraining: the guidance strength lam is selected on the validation
fold, then heart-rate control, fidelity and TSTR are reported on the test fold against lam = 0 with
the same seeds.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..data.signal_utils import heart_rate
from ..diffusion.physics_guidance import limit_cycle_energy
from ..diffusion.train import generate, make_model
from ..eval.fidelity import frechet_distance, log_spectral_distance, mmd_rbf
from ..eval.models import auroc, embed, predict, train_classifier
from .common import device, env_info, get_profile, load_splits, out_dir, save_json
from .run_diffusion import discriminative_score, sample_conditions
from .run_guidance import saturated_fraction

LAMS = (0.0, 0.05, 0.1, 0.2, 0.4)
REQ = (50.0, 70.0, 90.0, 110.0)


def hr_control(model, dev, n_per, seed, sampler, lam):
    rng = np.random.default_rng(seed)
    req = np.repeat(np.array(REQ, dtype=np.float32), n_per)
    lab = rng.integers(0, 2, len(req))
    x = generate(model, lab, req, dev, w=sampler["w"], seed=seed, spacing=sampler["spacing"],
                 threshold=sampler["threshold"], phys=lam)
    hr = np.array([heart_rate(r)[0] for r in x], dtype=np.float32)
    rrs = np.array([heart_rate(r)[1] for r in x], dtype=np.float32)
    ok = np.isfinite(hr)
    return {"mae_bpm": float(np.mean(np.abs(hr[ok] - req[ok]))) if ok.any() else float("nan"),
            "measured_mean_bpm": [float(np.nanmean(hr[req == r])) for r in REQ],
            "detectable_fraction": float(ok.mean()),
            "rr_std_ms_median": float(np.nanmedian(rrs) * 1e3)}


def energy(x, hr):
    with torch.no_grad():
        e = limit_cycle_energy(torch.tensor(x[:, None, :]), torch.tensor(hr, dtype=torch.float32))
    return float(e.median())


def main():
    prof = get_profile()
    od = out_dir(prof)
    dev = device()
    rng = np.random.default_rng(0)
    _, train, val, test = load_splits(prof)
    n_sweep = 300 if prof.name == "full" else 32
    n_final = 2000 if prof.name == "full" else 64
    n_hr = 40 if prof.name == "full" else 4

    import json
    gsel = json.loads((od / "guidance_results.json").read_text())["selected_w"]["diff_full"]
    sampler = {"spacing": gsel["spacing"], "threshold": gsel["threshold"], "w": gsel["w"]}
    model = make_model().to(dev)
    model.load_state_dict(torch.load(od / "ckpt_diff_full.pt", map_location=dev))

    feat, _ = train_classifier(train.x, train.y, val.x, val.y, dev, steps=prof.clf_steps)
    f_val, f_test = embed(feat, val.x, dev), embed(feat, test.x, dev)
    real_hr = np.array([heart_rate(r) for r in test.x], dtype=np.float32)
    res = {"env": env_info(), "sampler": sampler, "lams": list(LAMS), "sweep_on_validation": {},
           "real_test_rr_std_ms_median": float(np.nanmedian(real_hr[:, 1]) * 1e3)}

    y_s, hr_s = sample_conditions(train.y, train.hr, n_sweep, rng)
    for lam in LAMS:
        t0 = time.time()
        x = generate(model, y_s, hr_s, dev, w=sampler["w"], seed=21, spacing=sampler["spacing"],
                     threshold=sampler["threshold"], phys=lam)
        r = {"lam": lam, "saturated_fraction": saturated_fraction(x),
             "frechet_ecg_distance": frechet_distance(f_val, embed(feat, x, dev)),
             "log_spectral_distance_db": log_spectral_distance(val.x, x),
             "limit_cycle_energy_median": energy(x, hr_s),
             "hr_control": hr_control(model, dev, n_hr, 7, sampler, lam),
             "seconds": round(time.time() - t0, 1)}
        res["sweep_on_validation"][f"{lam:g}"] = r
        print(f"[physguide] lam={lam:<5g} | HR MAE {r['hr_control']['mae_bpm']:.2f} bpm | FED(val) "
              f"{r['frechet_ecg_distance']:.2f} | LSD {r['log_spectral_distance_db']:.2f} | "
              f"E_lc {r['limit_cycle_energy_median']:.3f} | RR std {r['hr_control']['rr_std_ms_median']:.0f} ms | "
              f"sat {r['saturated_fraction']:.3f}", flush=True)
        save_json(res, od / "physics_guidance_results.json")

    sw = res["sweep_on_validation"]
    fed0 = sw["0"]["frechet_ecg_distance"]
    ok = [r for r in sw.values() if r["frechet_ecg_distance"] <= 1.10 * fed0 and r["saturated_fraction"] < 0.05]
    best = min(ok, key=lambda r: r["hr_control"]["mae_bpm"])
    lam_sel = best["lam"]
    res["selected_lam"] = lam_sel
    res["selection_rule"] = "min HR-control MAE s.t. FED(val) <= 1.10 x FED(val, lam=0) and saturated < 5 %"
    print(f"[physguide] selected lam = {lam_sel:g}", flush=True)

    res["final_on_test"] = {}
    y_f, hr_f = sample_conditions(train.y, train.hr, n_final, rng)
    for tag, lam in (("unguided", 0.0), ("physics_guided", lam_sel)):
        x = generate(model, y_f, hr_f, dev, w=sampler["w"], seed=33, spacing=sampler["spacing"],
                     threshold=sampler["threshold"], phys=lam)
        f = embed(feat, x, dev)
        ds_, acc = discriminative_score(test.x, x, dev, steps=max(20, prof.clf_steps // 2))
        tstr, _ = train_classifier(x, y_f, val.x, val.y, dev, steps=prof.clf_steps)
        res["final_on_test"][tag] = {
            "lam": lam, "n": len(x), "saturated_fraction": saturated_fraction(x),
            "frechet_ecg_distance": frechet_distance(f_test, f), "mmd2_embedding": mmd_rbf(f_test, f),
            "log_spectral_distance_db": log_spectral_distance(test.x, x),
            "discriminative_score": ds_, "tstr_auroc": auroc(test.y, predict(tstr, test.x, dev)),
            "limit_cycle_energy_median": energy(x, hr_f),
            "hr_control": hr_control(model, dev, 2 * n_hr, 99, sampler, lam)}
        if tag == "physics_guided":
            np.savez_compressed(od / "synth_diff_full_physguided.npz", x=x, y=y_f)
        r = res["final_on_test"][tag]
        print(f"[physguide] test {tag:>14} (lam={lam:g}): HR MAE {r['hr_control']['mae_bpm']:.2f} bpm | FED "
              f"{r['frechet_ecg_distance']:.2f} | MMD {r['mmd2_embedding']:.4f} | LSD {r['log_spectral_distance_db']:.2f} | "
              f"disc {r['discriminative_score']:.3f} | TSTR {r['tstr_auroc']:.3f}", flush=True)
        save_json(res, od / "physics_guidance_results.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lams = [r["lam"] for r in sw.values()]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].plot(lams, [r["hr_control"]["mae_bpm"] for r in sw.values()], "o-", color="#2a78d6")
        ax[0].set_xlabel("physics guidance strength λ"); ax[0].set_ylabel("HR-control MAE [bpm]")
        ax[0].axvline(lam_sel, color="#888", ls=":"); ax[0].grid(alpha=0.3)
        ax[1].plot(lams, [r["frechet_ecg_distance"] for r in sw.values()], "o-", color="#eb6834")
        ax[1].set_xlabel("physics guidance strength λ"); ax[1].set_ylabel("FED (validation) ↓")
        ax[1].axvline(lam_sel, color="#888", ls=":"); ax[1].grid(alpha=0.3)
        fig.suptitle("Limit-cycle physics guidance (no retraining), selected on validation", fontsize=10)
        fig.tight_layout(); fig.savefig(od / "fig_physics_guidance.png", dpi=130)
    except Exception as e:
        print("[physguide] figure skipped:", e)


if __name__ == "__main__":
    main()
