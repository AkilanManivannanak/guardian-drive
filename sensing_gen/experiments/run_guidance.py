"""Step 6: diagnose and fix sample divergence in the diffusion models.

The first full run showed ~40 % of samples saturating at the x0-clip boundary. A guidance sweep showed
the failure persists at w = 0, so classifier-free guidance is not the cause. The cause is the sampler:
DDIM started at t = T-1, where the cosine schedule has alpha_bar ~ 2.4e-9, so the x0 estimate
amplifies epsilon errors ~2e4x, and a static clip then pins them to the rail.

This step sweeps sampler (trailing/static = original, leading/static, leading/dynamic) x guidance w on
the *validation* fold, selects by Frechet distance among configurations with < 5 % saturated samples,
regenerates the synthetic sets, re-scores fidelity on the test fold and reruns the diffusion-based
downstream configurations. No retraining is involved.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ..data.datasets import NoiseBank, corrupt_at_snr
from ..diffusion.train import generate, make_model
from ..eval.fidelity import frechet_distance, log_spectral_distance, mmd_rbf
from ..eval.models import auroc, embed, predict, train_classifier
from ..physics.sensor import ElectrodeSkinSensor
from .common import DATA, device, env_info, get_profile, load_splits, out_dir, save_json
from .run_diffusion import discriminative_score, sample_conditions

WS = (0.0, 1.0, 2.0)
SAMPLERS = (("trailing", "static"), ("leading", "static"), ("leading", "dynamic"))


def saturated_fraction(x: np.ndarray, clip: float = 12.0) -> float:
    return float(((np.abs(x) >= clip - 0.1).mean(1) > 0.005).mean())


def main():
    prof = get_profile()
    od = out_dir(prof)
    dev = device()
    rng = np.random.default_rng(0)
    _, train, val, test = load_splits(prof)
    low_tr = train.subset(np.load(od / "low10_indices.npy"))
    n_sweep = 300 if prof.name == "full" else 32
    n_final = 2000 if prof.name == "full" else 64

    models = {}
    for k in ("diff_full", "diff_simpre_low10"):
        m = make_model().to(dev)
        m.load_state_dict(torch.load(od / f"ckpt_{k}.pt", map_location=dev))
        models[k] = m
    cond = {"diff_full": train, "diff_simpre_low10": low_tr}

    feat, _ = train_classifier(train.x, train.y, val.x, val.y, dev, steps=prof.clf_steps)
    f_val, f_test = embed(feat, val.x, dev), embed(feat, test.x, dev)
    res = {"env": env_info(), "sweep_on_validation": {}, "selected_w": {}, "final_on_test": {}}

    # 1) sweep on validation
    for k, m in models.items():
        res["sweep_on_validation"][k] = {}
        y_s, hr_s = sample_conditions(cond[k].y, cond[k].hr, n_sweep, rng)
        for sp, th in SAMPLERS:
            for w in WS:
                t0 = time.time()
                x = generate(m, y_s, hr_s, dev, w=w, seed=21, spacing=sp, threshold=th)
                r = {"spacing": sp, "threshold": th, "w": w, "saturated_fraction": saturated_fraction(x),
                     "frechet_ecg_distance": frechet_distance(f_val, embed(feat, x, dev)),
                     "log_spectral_distance_db": log_spectral_distance(val.x, x),
                     "seconds": round(time.time() - t0, 1)}
                res["sweep_on_validation"][k][f"{sp}/{th}/w={w:g}"] = r
                print(f"[sampler] {k:>18} {sp:>8}/{th:<7} w={w:<3} | saturated {r['saturated_fraction']:.3f} | "
                      f"FED(val) {r['frechet_ecg_distance']:.2f} | LSD {r['log_spectral_distance_db']:.2f}", flush=True)
        cands = [r for r in res["sweep_on_validation"][k].values() if r["saturated_fraction"] < 0.05]
        cands = cands or list(res["sweep_on_validation"][k].values())
        best = min(cands, key=lambda r: r["frechet_ecg_distance"])
        res["selected_w"][k] = {"spacing": best["spacing"], "threshold": best["threshold"], "w": best["w"]}
        save_json(res, od / "guidance_results.json")

    # 2) regenerate with the selected guidance + 3) fidelity on test
    synth = {}
    for k, m in models.items():
        sel = res["selected_w"][k]
        w = sel["w"]
        y_s, hr_s = sample_conditions(cond[k].y, cond[k].hr, n_final, rng)
        x = generate(m, y_s, hr_s, dev, w=w, seed=33, spacing=sel["spacing"], threshold=sel["threshold"])
        synth[k] = (x, y_s)
        np.savez_compressed(od / f"synth_{k}_wsel.npz", x=x, y=y_s)
        f = embed(feat, x, dev)
        ds_, acc = discriminative_score(test.x, x, dev, steps=max(20, prof.clf_steps // 2))
        tstr, _ = train_classifier(x, y_s, val.x, val.y, dev, steps=prof.clf_steps)
        res["final_on_test"][k] = {
            "w": w, "sampler": f"{sel['spacing']}/{sel['threshold']}", "n": len(x), "saturated_fraction": saturated_fraction(x),
            "frechet_ecg_distance": frechet_distance(f_test, f), "mmd2_embedding": mmd_rbf(f_test, f),
            "log_spectral_distance_db": log_spectral_distance(test.x, x),
            "discriminative_score": ds_, "discriminator_accuracy": acc,
            "tstr_auroc": auroc(test.y, predict(tstr, test.x, dev)),
        }
        print(f"[sampler] final {k} ({sel}): " + "  ".join(
            f"{a}={b:.3f}" for a, b in res["final_on_test"][k].items() if isinstance(b, float)), flush=True)
        save_json(res, od / "guidance_results.json")

    # 4) downstream with the regenerated sets
    nb = NoiseBank(DATA / "nstdb")
    sensor = ElectrodeSkinSensor()
    test_sets = {"clean": test.x}
    for snr in (6, 0):
        noise = nb.sample(len(test), test.x.shape[1], "test", np.random.default_rng(1000 + snr))
        test_sets[f"nstdb_{snr}dB"] = corrupt_at_snr(test.x, noise, snr)
    configs = {
        "10pct_real": (low_tr, {"real+sim_pretrained_diffusion(w_sel)": {"extra": synth["diff_simpre_low10"]},
                                "real+sim_pretrained_diffusion(w_sel)+sensor_model_aug":
                                    {"extra": synth["diff_simpre_low10"], "augment": sensor}}),
        "100pct_real": (train, {"real+diffusion(w_sel)": {"extra": synth["diff_full"]},
                                "real+diffusion(w_sel)+sensor_model_aug": {"extra": synth["diff_full"],
                                                                           "augment": sensor}}),
    }
    res["downstream"] = {}
    for reg, (tr, cfgs) in configs.items():
        res["downstream"][reg] = {}
        for name, kw in cfgs.items():
            runs = []
            for seed in prof.seeds:
                m, _ = train_classifier(tr.x, tr.y, val.x, val.y, dev, steps=prof.clf_steps, seed=seed,
                                        extra=kw.get("extra"), augment=kw.get("augment"))
                runs.append({k: auroc(test.y, predict(m, v, dev)) for k, v in test_sets.items()})
            summ = {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
                    for k in test_sets}
            res["downstream"][reg][name] = {"auroc": summ, "runs": runs}
            print(f"[guidance] {reg:>11} | {name:<55} | " +
                  " | ".join(f"{k} {v['mean']:.3f}±{v['std']:.3f}" for k, v in summ.items()), flush=True)
            save_json(res, od / "guidance_results.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = [("real (PTB-XL test)", test.x, test.y),
                ("diffusion, full real (fixed sampler)", *synth["diff_full"]),
                ("diffusion, sim-pretrained + 10% real (fixed sampler)", *synth["diff_simpre_low10"])]
        fig, ax = plt.subplots(len(rows), 4, figsize=(16, 7), sharex=True)
        t = np.arange(1000) / 100
        r2 = np.random.default_rng(5)
        for r, (name, x, y) in enumerate(rows):
            for c in range(4):
                cls = c // 2
                i = r2.choice(np.nonzero(y == cls)[0])  # random, not cherry-picked
                ax[r, c].plot(t, x[i], lw=0.7, color="#222" if r == 0 else "#d6336c")
                ax[r, c].set_title(f"{name} · {'NORM' if cls == 0 else 'ABN'}", fontsize=8)
                ax[r, c].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(od / "fig_samples_wsel.png", dpi=120)
    except Exception as e:
        print("[guidance] figure skipped:", e)


if __name__ == "__main__":
    main()
