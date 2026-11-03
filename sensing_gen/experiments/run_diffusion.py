"""Step 3: conditional diffusion models (real-only vs simulation-pretrained) + fidelity evaluation."""
from __future__ import annotations

import time

import numpy as np
import torch

from ..diffusion.train import generate, train_diffusion
from ..eval.fidelity import (frechet_distance, hr_stats, hr_wasserstein, log_spectral_distance,
                             mmd_rbf)
from ..eval.models import auroc, embed, predict, train_classifier
from ..physics.calibrate import ParamBank, sample_from_bank
from .common import device, env_info, get_profile, load_splits, out_dir, save_json


def sample_conditions(y_ref, hr_ref, n, rng):
    idx = rng.integers(0, len(y_ref), n)
    hr = np.asarray(hr_ref, dtype=np.float32)[idx]
    hr = np.where(np.isfinite(hr), hr, np.nanmedian(hr_ref))
    return np.asarray(y_ref)[idx].astype(np.int64), hr.astype(np.float32)


def discriminative_score(real, fake, dev, steps, seed=0):
    """|accuracy - 0.5| of a classifier trained to tell real from synthetic (0 = indistinguishable)."""
    rng = np.random.default_rng(seed)
    n = min(len(real), len(fake))
    r, f = real[rng.permutation(len(real))[:n]], fake[rng.permutation(len(fake))[:n]]
    h = n // 2
    xtr = np.concatenate([r[:h], f[:h]]); ytr = np.r_[np.zeros(h), np.ones(h)].astype(np.int64)
    xte = np.concatenate([r[h:], f[h:]]); yte = np.r_[np.zeros(n - h), np.ones(n - h)].astype(np.int64)
    m, _ = train_classifier(xtr, ytr, xte, yte, dev, steps=steps, seed=seed, eval_every=steps)
    acc = float(np.mean((predict(m, xte, dev) > 0.5) == yte))
    return abs(acc - 0.5), acc


def main():
    prof = get_profile()
    od = out_dir(prof)
    dev = device()
    rng = np.random.default_rng(0)
    ds, train, val, test = load_splits(prof)
    low = np.load(od / "low10_indices.npy")
    low_tr = train.subset(low)
    bank_low = ParamBank.load(od / "param_bank_low10.npz")
    bank_full = ParamBank.load(od / "param_bank_full.npz")
    res = {"env": env_info(), "train_logs": {}}
    timings = {}

    # ---------------- simulator data (also used to pre-train the low-data diffusion model) -------
    n_pre = 8000 if prof.name == "full" else 256
    y_pre, _ = sample_conditions(low_tr.y, low_tr.hr, n_pre, rng)
    t0 = time.time()
    x_pre = sample_from_bank(bank_low, y_pre, rng, seed=1)
    timings["simulate_pretrain_set_s"] = round(time.time() - t0, 1)
    hr_pre = hr_stats(x_pre)
    print(f"[diffusion] simulated {n_pre} pre-training records in {timings['simulate_pretrain_set_s']} s")

    models = {}
    t0 = time.time()
    models["diff_full"], res["train_logs"]["diff_full"] = train_diffusion(
        train.x, train.y, train.hr, prof.diff_steps_full, dev, tag=":full")
    timings["train_full_s"] = round(time.time() - t0, 1)
    models["diff_low10"], res["train_logs"]["diff_low10"] = train_diffusion(
        low_tr.x, low_tr.y, low_tr.hr, prof.diff_steps_low, dev, tag=":low10")
    pre, res["train_logs"]["sim_pretrain"] = train_diffusion(
        x_pre, y_pre, hr_pre, prof.diff_steps_sim, dev, tag=":sim-pretrain")
    models["diff_simpre_low10"], res["train_logs"]["diff_simpre_low10"] = train_diffusion(
        low_tr.x, low_tr.y, low_tr.hr, prof.diff_steps_low, dev, init=pre.state_dict(), lr=1e-4,
        tag=":sim->low10")
    for k, m in models.items():
        torch.save(m.state_dict(), od / f"ckpt_{k}.pt")

    # ---------------- synthetic sets -----------------------------------------------------------
    n = prof.n_synth
    synth = {}
    cond = {"diff_full": train, "diff_low10": low_tr, "diff_simpre_low10": low_tr}
    for k, m in models.items():
        y_s, hr_s = sample_conditions(cond[k].y, cond[k].hr, n, rng)
        t0 = time.time()
        synth[k] = (generate(m, y_s, hr_s, dev, seed=3), y_s)
        timings[f"sample_{k}_s"] = round(time.time() - t0, 1)
    for k, bank, ref in (("sim_low10", bank_low, low_tr), ("sim_full", bank_full, train)):
        y_s, _ = sample_conditions(ref.y, ref.hr, n, rng)
        synth[k] = (sample_from_bank(bank, y_s, rng, seed=5), y_s)
    for k, (x, y) in synth.items():
        np.savez_compressed(od / f"synth_{k}.npz", x=x, y=y)
    ntr = min(len(train), n)
    sub = rng.choice(len(train), ntr, replace=False)
    synth_eval = {"real_train_split": (train.x[sub], train.y[sub]), **synth}

    # ---------------- fidelity -----------------------------------------------------------------
    feat_model, _ = train_classifier(train.x, train.y, val.x, val.y, dev, steps=prof.clf_steps)
    f_test = embed(feat_model, test.x, dev)
    fid = {}
    for k, (x, y) in synth_eval.items():
        f = embed(feat_model, x, dev)
        ds_, acc = discriminative_score(test.x, x, dev, steps=max(20, prof.clf_steps // 2))
        tstr, _ = train_classifier(x, y, val.x, val.y, dev, steps=prof.clf_steps)
        fid[k] = {
            "frechet_ecg_distance": frechet_distance(f_test, f),
            "mmd2_embedding": mmd_rbf(f_test, f),
            "log_spectral_distance_db": log_spectral_distance(test.x, x),
            "hr_wasserstein_bpm": hr_wasserstein(test.x, x),
            "discriminative_score": ds_, "discriminator_accuracy": acc,
            "tstr_auroc": auroc(test.y, predict(tstr, test.x, dev)),
        }
        print(f"[fidelity] {k:>20}: " + "  ".join(f"{a}={b:.3f}" for a, b in fid[k].items()), flush=True)
    res["fidelity_vs_real_test"] = fid
    res["trtr_auroc_reference"] = auroc(test.y, predict(feat_model, test.x, dev))

    # ---------------- heart-rate controllability -------------------------------------------------
    req = np.repeat(np.array([50, 70, 90, 110], dtype=np.float32), max(8, n // 32))
    lab = rng.integers(0, 2, len(req))
    xs = generate(models["diff_full"], lab, req, dev, seed=11)
    meas = hr_stats(xs)
    ok = np.isfinite(meas)
    res["hr_control"] = {
        "requested_bpm": [50, 70, 90, 110],
        "measured_mean_bpm": [float(np.nanmean(meas[req == r])) for r in (50, 70, 90, 110)],
        "mae_bpm": float(np.mean(np.abs(meas[ok] - req[ok]))),
        "detectable_fraction": float(ok.mean()),
    }
    print("[diffusion] HR control:", res["hr_control"], flush=True)
    res["timings"] = timings
    save_json(res, od / "diffusion_results.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = [("real (PTB-XL test)", test.x, test.y), ("physics simulator (calibrated)", *synth["sim_full"]),
                ("diffusion (full real)", *synth["diff_full"]),
                ("diffusion (sim-pretrained, 10% real)", *synth["diff_simpre_low10"])]
        fig, ax = plt.subplots(len(rows), 2, figsize=(13, 9), sharex=True)
        t = np.arange(1000) / 100
        for r, (name, x, y) in enumerate(rows):
            for c in (0, 1):
                i = np.nonzero(y == c)[0][0]
                ax[r, c].plot(t, x[i], lw=0.8, color="#222" if r == 0 else "#d6336c")
                ax[r, c].set_title(f"{name} - {'NORM' if c == 0 else 'ABNORMAL'}", fontsize=9)
                ax[r, c].grid(alpha=0.3)
        ax[-1, 0].set_xlabel("time [s]"); ax[-1, 1].set_xlabel("time [s]")
        fig.tight_layout()
        fig.savefig(od / "fig_samples.png", dpi=120)
    except Exception as e:
        print("[diffusion] figure skipped:", e)


if __name__ == "__main__":
    main()
