"""Step 13: how much real data does simulation / generation replace?

The question behind "scale training beyond real-world collection", measured directly:

  * real-only reference curve: AUROC vs fraction of real PTB-XL training data (10, 25, 50, 100 %;
    nested subsets that all contain the 10 % set),
  * synthetic scaling at 10 % real: AUROC vs number of synthetic records N (0 ... 8000) from
      - the calibrated physics simulator (parameter bank fitted on the 10 % subset only), and
      - the simulation-pretrained diffusion model fine-tuned on the 10 % subset only,
    so no generator has seen data outside the 10 % regime.
  * real-data equivalence: the real fraction whose (log-interpolated) real-only AUROC equals each
    synthetic configuration's AUROC.

Every configuration uses the physics sensor-model augmentation (the best pipeline from step 4), the same
classifier, steps and seeds; test conditions are clean and real NSTDB noise at 0 dB (held-out half).
"""
from __future__ import annotations

import time

import numpy as np

from ..data.datasets import NoiseBank, corrupt_at_snr
from ..diffusion.train import generate, make_model
from ..eval.models import auroc, predict, train_classifier
from ..physics.calibrate import ParamBank, sample_from_bank
from ..physics.sensor import ElectrodeSkinSensor
from .common import DATA, device, env_info, get_profile, load_splits, out_dir, save_json
from .run_diffusion import sample_conditions

FRACS = (0.10, 0.25, 0.50, 1.00)


def equivalent_fraction(auc: float, fracs, aucs) -> float:
    """Invert the monotone envelope of the real-only curve on a log-fraction axis (clipped to range)."""
    lf = np.log(np.asarray(fracs))
    a = np.maximum.accumulate(np.asarray(aucs))
    if auc <= a[0]:
        return float(fracs[0])
    if auc >= a[-1]:
        return float(fracs[-1])
    return float(np.exp(np.interp(auc, a, lf)))


def main():
    import torch
    prof = get_profile()
    od = out_dir(prof)
    dev = device()
    full = prof.name == "full"
    rng = np.random.default_rng(0)
    _, train, val, test = load_splits(prof)
    low = np.load(od / "low10_indices.npy")
    sizes = (500, 1000, 2000, 4000, 8000) if full else (64, 128)
    sensor = ElectrodeSkinSensor()
    nb = NoiseBank(DATA / "nstdb")
    noise = nb.sample(len(test), test.x.shape[1], "test", np.random.default_rng(1000))
    test_sets = {"clean": test.x, "nstdb_0dB": corrupt_at_snr(test.x, noise, 0)}

    # nested real subsets: 10 % set first, then a fixed random order of the rest
    rest = np.setdiff1d(np.arange(len(train)), low)
    order = np.concatenate([low, rng.permutation(rest)])

    def run(tr, extra=None):
        runs = []
        for seed in prof.seeds:
            m, _ = train_classifier(tr.x, tr.y, val.x, val.y, dev, steps=prof.clf_steps, seed=seed,
                                    extra=extra, augment=sensor, p_extra=0.5)
            runs.append({k: auroc(test.y, predict(m, v, dev)) for k, v in test_sets.items()})
        return {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
                for k in test_sets} | {"runs": runs}

    res = {"env": env_info(), "fractions": list(FRACS), "sizes": list(sizes), "real_only": {}, "synthetic": {}}
    path = od / "scaling_results.json"
    for f in FRACS:
        n = len(low) if f == FRACS[0] else int(round(f * len(train)))
        t0 = time.time()
        res["real_only"][f"{f:g}"] = run(train.subset(order[:n])) | {"n_real": n}
        r = res["real_only"][f"{f:g}"]
        print(f"[scaling] real-only {100 * f:5.1f} % (n={n}) | clean {r['clean']['mean']:.3f} | 0 dB "
              f"{r['nstdb_0dB']['mean']:.3f} | {time.time() - t0:.0f} s", flush=True)
        save_json(res, path)

    low_tr = train.subset(low)
    nmax = max(sizes)
    y_s, hr_s = sample_conditions(low_tr.y, low_tr.hr, nmax, rng)
    bank = ParamBank.load(od / "param_bank_low10.npz")
    pools = {"simulator": (sample_from_bank(bank, y_s, rng, seed=77), y_s)}
    diff = make_model().to(dev)
    diff.load_state_dict(torch.load(od / "ckpt_diff_simpre_low10.pt", map_location=dev))
    import json
    gsel = json.loads((od / "guidance_results.json").read_text())["selected_w"]["diff_simpre_low10"]
    t0 = time.time()
    xd = generate(diff, y_s, hr_s, dev, w=gsel["w"], seed=78, spacing=gsel["spacing"], threshold=gsel["threshold"])
    print(f"[scaling] generated {nmax} diffusion records in {time.time() - t0:.0f} s", flush=True)
    pools["sim_pretrained_diffusion"] = (xd, y_s)

    fr = list(FRACS)
    for k in ("clean", "nstdb_0dB"):
        res.setdefault("real_curve", {})[k] = [res["real_only"][f"{f:g}"][k]["mean"] for f in FRACS]
    for name, (xp, yp) in pools.items():
        res["synthetic"][name] = {}
        for n in sizes:
            t0 = time.time()
            r = run(low_tr, extra=(xp[:n], yp[:n]))
            r["equivalent_real_fraction"] = {k: equivalent_fraction(r[k]["mean"], fr, res["real_curve"][k])
                                             for k in ("clean", "nstdb_0dB")}
            res["synthetic"][name][str(n)] = r
            print(f"[scaling] 10 % real + {n:>5} {name:<24} | clean {r['clean']['mean']:.3f} "
                  f"(≈{100 * r['equivalent_real_fraction']['clean']:.0f} % real) | 0 dB {r['nstdb_0dB']['mean']:.3f} "
                  f"(≈{100 * r['equivalent_real_fraction']['nstdb_0dB']:.0f} % real) | {time.time() - t0:.0f} s", flush=True)
            save_json(res, path)

    plot(res, od / "fig_scaling.png")


def plot(res: dict, path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sizes = res["sizes"]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        for i, k in enumerate(("clean", "nstdb_0dB")):
            for f, a, ls in zip(res["fractions"], res["real_curve"][k], (":", "-.", "--", "-")):
                ax[i].axhline(a, color="#777", lw=1, ls=ls, label=f"{100 * f:.0f} % real only")
            for name, col, lab in (("simulator", "#2b9e5e", "10 % real + physics simulator"),
                                   ("sim_pretrained_diffusion", "#d6336c", "10 % real + diffusion (trained on the same 10 %)")):
                m = [res["synthetic"][name][str(n)][k]["mean"] for n in sizes]
                sd = [res["synthetic"][name][str(n)][k]["std"] for n in sizes]
                ax[i].errorbar(sizes, m, yerr=sd, fmt="o-", color=col, capsize=3, label=lab)
            ax[i].set_xscale("log"); ax[i].set_xlabel("synthetic records added (log scale)")
            ax[i].set_ylabel("AUROC (mean ± std, 3 seeds)"); ax[i].grid(alpha=0.3)
            ax[i].set_title("clean test set" if k == "clean" else "test set + real NSTDB noise at 0 dB", fontsize=10)
        h, l = ax[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8)
        fig.suptitle("How much real data does synthetic data replace? (sensor-model augmentation in every run)", fontsize=10)
        fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(path, dpi=130)
    except Exception as e:
        print("[scaling] figure skipped:", e)

if __name__ == "__main__":
    main()
