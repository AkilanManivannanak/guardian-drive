"""Step 7: real in-car ECG. Do models trained with simulation stay calm on real driving data?

Data: PhysioNet "Stress Recognition in Automobile Drivers" (drivedb; Healey & Picard, IEEE T-ITS
2005). ECG was recorded from volunteer drivers on real roads around Boston. No cardiac diagnoses
are given, so every 10-s window is treated as a NORMAL ECG, and any "abnormal" output counts as a
**false alarm**. The question this answers is the deployment one: how often would each model raise
a spurious cardiac alert while someone is driving?

Protocol
* Classifiers: the same 1-D ResNet as run_downstream, trained on 100 % of PTB-XL folds 1-8 (3 seeds)
  with: real only / + physics sensor model / + real NSTDB noise (reference) / + diffusion samples
  (fixed sampler) + sensor model.
* Operating point: for each model, the threshold giving 90 % specificity on PTB-XL validation NORM
  records (fold 9). Nothing is tuned on driving data.
* drivedb ECG is resampled to 100 Hz, band-passed 0.5-40 Hz like PTB-XL, and amplitude-matched per
  record (median window std = PTB-XL training median), because chest-lead placement differs from
  lead II.
* Windows are split into "start-of-record (rest, approx.)" = first 10 min, and "driving" = the rest.
  The split is approximate: drivedb's protocol starts with a rest period, but its marker channel is
  not used.
"""
from __future__ import annotations

import time
from fractions import Fraction

import numpy as np
from scipy import signal as sps

from ..data.signal_utils import bandpass, detect_rpeaks
from ..data.wfdb_io import read_record_multi
from ..data.datasets import NoiseBank
from ..eval.models import auroc, predict, train_classifier
from ..physics.sensor import ElectrodeSkinSensor
from .common import DATA, device, env_info, get_profile, load_splits, out_dir, save_json

FS = 100.0
WIN = 1000


def load_drivedb(root, ref_std: float, scale_mv: float):
    """Return list of dicts: record, windows (n, 1000), is_rest (n,), quality (n,)."""
    recs = [r.strip() for r in (root / "RECORDS").read_text().split() if r.strip()]
    out = []
    for r in recs:
        try:
            sigs, hdr = read_record_multi(root / r)
        except Exception as e:  # keep going; report skipped records
            print(f"[driving] skip {r}: {e}")
            continue
        ecg_key = next((k for k in sigs if "ECG" in k.upper()), None)
        if ecg_key is None:
            print(f"[driving] skip {r}: no ECG channel ({list(sigs)})")
            continue
        x, fs = sigs[ecg_key]
        frac = Fraction(FS / fs).limit_denominator(1000)
        x = sps.resample_poly(x.astype(np.float64), frac.numerator, frac.denominator)
        x = bandpass(np.nan_to_num(x), FS) / scale_mv
        n = len(x) // WIN
        if n < 6:
            continue
        w = x[: n * WIN].reshape(n, WIN)
        w = w - np.median(w, axis=1, keepdims=True)
        w = w * (ref_std / (np.median(w.std(1)) + 1e-8))      # per-record amplitude match
        w = np.clip(w, -12, 12).astype(np.float32)
        t0 = np.arange(n) * WIN / FS
        q = np.array([4 <= len(detect_rpeaks(v)) <= 30 for v in w])
        out.append({"record": r, "ecg_channel": ecg_key, "fs": fs, "x": w, "is_rest": t0 < 600, "ok": q})
        print(f"[driving] {r}: {n} windows ({fs:g} Hz ECG '{ecg_key}'), detectable beats {q.mean():.2f}", flush=True)
    return out


def main():
    prof = get_profile()
    od = out_dir(prof)
    dev = device()
    _, train, val, test = load_splits(prof)
    ref_std = float(np.median(train.x.std(1)))
    drives = load_drivedb(DATA / "drivedb", ref_std, scale_mv=1.0)
    if not drives:
        raise SystemExit("no drivedb records found; run: python -m sensing_gen.data.download --drivedb-only")
    X = np.concatenate([d["x"] for d in drives])
    rest = np.concatenate([d["is_rest"] for d in drives])
    ok = np.concatenate([d["ok"] for d in drives])
    rec_id = np.concatenate([[i] * len(d["x"]) for i, d in enumerate(drives)])

    nb = NoiseBank(DATA / "nstdb")
    sensor = ElectrodeSkinSensor()

    def nstdb_aug(x, gen):
        import torch
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen))
        r = np.random.default_rng(seed)
        noise = nb.sample(x.shape[0], x.shape[1], "train", r)
        snr = r.uniform(-3, 18, (x.shape[0], 1))
        ps = (x.numpy() ** 2).mean(1, keepdims=True)
        pn = (noise**2).mean(1, keepdims=True) + 1e-12
        return torch.tensor(x.numpy() + noise * np.sqrt(ps / (pn * 10 ** (snr / 10))), dtype=torch.float32)

    synth_path = od / "synth_diff_full_wsel.npz"
    configs = {"real": {}, "real+sensor_model_aug": {"augment": sensor},
               "real+nstdb_noise_aug(reference)": {"augment": nstdb_aug}}
    if synth_path.exists():
        z = np.load(synth_path)
        configs["real+diffusion(fixed)+sensor_model_aug"] = {"extra": (z["x"], z["y"]), "augment": sensor}

    val_norm = val.x[val.y == 0]
    test_norm = test.x[test.y == 0]
    res = {"env": env_info(), "n_records": len(drives), "n_windows": int(len(X)),
           "n_windows_rest": int(rest.sum()), "frac_windows_detectable_beats": float(ok.mean()),
           "records": [{k: (v if not isinstance(v, np.ndarray) else None) for k, v in d.items()
                        if k in ("record", "ecg_channel", "fs")} | {"n_windows": int(len(d["x"]))}
                       for d in drives],
           "results": {}}
    rng = np.random.default_rng(0)
    per_cfg_flags = {}
    for name, kw in configs.items():
        t0, runs, flags_all = time.time(), [], []
        for seed in prof.seeds:
            m, _ = train_classifier(train.x, train.y, val.x, val.y, dev, steps=prof.clf_steps, seed=seed,
                                    extra=kw.get("extra"), augment=kw.get("augment"))
            thr = float(np.quantile(predict(m, val_norm, dev), 0.90))  # 90 % specificity on validation NORM
            p_drive = predict(m, X, dev)
            flag = p_drive > thr
            flags_all.append(flag)
            per_rec = [float(flag[rec_id == i].mean()) for i in range(len(drives))]
            runs.append({
                "threshold": thr,
                "ptbxl_test_auroc": auroc(test.y, predict(m, test.x, dev)),
                "ptbxl_test_norm_false_alarm": float((predict(m, test_norm, dev) > thr).mean()),
                "drive_false_alarm_all": float(flag.mean()),
                "drive_false_alarm_rest": float(flag[rest].mean()) if rest.any() else None,
                "drive_false_alarm_driving": float(flag[~rest].mean()),
                "drive_false_alarm_detectable_only": float(flag[ok].mean()),
                "drive_false_alarm_median_per_driver": float(np.median(per_rec)),
                "drive_mean_abnormal_prob": float(p_drive.mean()),
            })
        per_cfg_flags[name] = np.mean(flags_all, axis=0)
        keys = [k for k in runs[0] if isinstance(runs[0][k], float)]
        summ = {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
                for k in keys}
        res["results"][name] = {"summary": summ, "runs": runs, "seconds": round(time.time() - t0, 1)}
        print(f"[driving] {name:<42} | PTB-XL NORM FA {summ['ptbxl_test_norm_false_alarm']['mean']:.3f} | "
              f"in-car FA all {summ['drive_false_alarm_all']['mean']:.3f}±{summ['drive_false_alarm_all']['std']:.3f}"
              f" | rest {summ.get('drive_false_alarm_rest', {}).get('mean', float('nan')):.3f}"
              f" | driving {summ['drive_false_alarm_driving']['mean']:.3f}", flush=True)
        save_json(res, od / "driving_results.json")

    # driver-level bootstrap CI for the change in false-alarm rate vs real-only
    base = per_cfg_flags["real"]
    res["bootstrap_vs_real"] = {}
    for name, f in per_cfg_flags.items():
        if name == "real":
            continue
        diffs = []
        for _ in range(2000):
            pick = rng.integers(0, len(drives), len(drives))
            m = np.concatenate([np.nonzero(rec_id == i)[0] for i in pick])
            diffs.append(f[m].mean() - base[m].mean())
        res["bootstrap_vs_real"][name] = {"delta_false_alarm": float(f.mean() - base.mean()),
                                          "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
        print(f"[driving] {name}: delta false-alarm vs real {f.mean() - base.mean():+.3f} "
              f"(95% driver-bootstrap CI {res['bootstrap_vs_real'][name]['ci95'][0]:+.3f} to "
              f"{res['bootstrap_vs_real'][name]['ci95'][1]:+.3f})", flush=True)
    save_json(res, od / "driving_results.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
        i = int(np.nonzero(~rest & ok)[0][len(np.nonzero(~rest & ok)[0]) // 2])
        ax[0].plot(np.arange(WIN) / FS, X[i], color="#222", lw=0.8)
        ax[0].set_title(f"Real in-car ECG while driving ({drives[rec_id[i]]['record']}, PhysioNet drivedb)", fontsize=10)
        ax[0].set_xlabel("time [s]"); ax[0].grid(alpha=0.3)
        names = list(res["results"])
        fa = [res["results"][n]["summary"]["drive_false_alarm_driving"]["mean"] for n in names]
        sd = [res["results"][n]["summary"]["drive_false_alarm_driving"]["std"] for n in names]
        ax[1].barh(range(len(names)), fa, xerr=sd, color=["#868e96", "#d6336c", "#1971c2", "#7048e8"][: len(names)])
        ax[1].axvline(0.10, color="k", ls="--", lw=0.8, label="design rate (10 %)")
        ax[1].set_yticks(range(len(names)), [n.replace("_", " ") for n in names], fontsize=8)
        ax[1].set_xlabel("false cardiac alarms while driving"); ax[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(od / "fig_driving.png", dpi=130)
    except Exception as e:
        print("[driving] figure skipped:", e)


if __name__ == "__main__":
    main()
