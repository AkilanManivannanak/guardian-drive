"""Step 15: human-behaviour signals. Activity recognition from a body-worn phone's IMU (UCI HAR).

Same question as the ECG work, on a second modality: do a physics sensor model and a conditional
diffusion generator make a classifier better and more robust, measured on real held-out data?

Protocol
* Data: UCI HAR raw inertial signals, 6 channels (total acceleration incl. gravity, angular rate),
  128-sample windows at 50 Hz, 6 activities. Official subject-disjoint split: train subjects
  (4 held out for validation / model selection) and 9 test subjects never seen in training.
* Test conditions: (a) test subjects as recorded; (b) the same windows from a re-oriented phone: each
  window rotated by a random uniform 3-D rotation applied to accelerometer and gyroscope together. A
  rotated rigid phone measures exactly R·a and R·ω, so (b) is a physical identity, not a noise model.
* Regimes: 100 % and 10 % of training windows. Each diffusion model is trained only on its regime's
  windows.
* Configurations (3 seeds): real; + physics IMU sensor model (orientation, calibration, loose mount,
  clock mismatch); + orientation-only augmentation (ablation); + diffusion samples; + diffusion and
  sensor model. Metric: macro-F1 (and accuracy).
* Generator fidelity: Frechet distance in a real-trained classifier's embedding, label consistency
  (accuracy of the real-trained classifier on synthetic windows), and TSTR macro-F1.
"""
from __future__ import annotations

import time

import numpy as np

from ..behavior.har_data import ACTIVITIES, load
from ..behavior.imu_sensor import IMUConfig, IMUSensorModel, random_rotation, rotate
from ..behavior.train import IMUDiffusion, embed, macro_f1, predict_logits, train_har
from ..eval.fidelity import frechet_distance
from .common import DATA, device, env_info, get_profile, out_dir, save_json

K = len(ACTIVITIES)


def main():
    prof = get_profile()
    od = out_dir(prof)
    dev = device()
    full = prof.name == "full"
    d = load(DATA / "har")
    xtr_all, ytr_all, str_all = d["train"]
    xte, yte, _ = d["test"]
    subj = np.unique(str_all)
    val_subj = subj[-4:]
    vm = np.isin(str_all, val_subj)
    xtr, ytr, xv, yv = xtr_all[~vm], ytr_all[~vm], xtr_all[vm], ytr_all[vm]
    if not full:
        rs = np.random.default_rng(0)
        i = rs.permutation(len(xtr))[:800]; xtr, ytr = xtr[i], ytr[i]
        i = rs.permutation(len(xte))[:400]; xte, yte = xte[i], yte[i]
    xte_rot = rotate(xte, random_rotation(np.random.default_rng(123), len(xte)))
    tests = {"test_as_recorded": xte, "test_reoriented_phone": xte_rot}
    steps_clf = 3000 if full else 150
    steps_diff = (6000, 3000) if full else (100, 100)
    print(f"[behavior] train {len(xtr)} windows ({len(np.unique(str_all[~vm]))} subjects), val {len(xv)} "
          f"({len(val_subj)} subjects), test {len(xte)} (9 unseen subjects)", flush=True)

    low = np.random.default_rng(1).permutation(len(xtr))[: len(xtr) // 10]
    regimes = {"100pct": (xtr, ytr), "10pct": (xtr[low], ytr[low])}
    imu = IMUSensorModel()
    rot_only = IMUSensorModel(IMUConfig(acc_scale=0, acc_bias_g=0, gyro_bias=0, p_wobble=0, p_warp=0,
                                        acc_noise_g=0, gyro_noise=0))
    res = {"env": env_info(), "activities": ACTIVITIES, "n_train": len(xtr), "n_val": len(xv), "n_test": len(xte),
           "fidelity": {}, "results": {}}
    path = od / "behavior_results.json"

    ref, _ = train_har(xtr, ytr, xv, yv, dev, steps=steps_clf, seed=0)
    f_test = embed(ref, xte, dev)
    for reg, (xr, yr) in regimes.items():
        t0 = time.time()
        g = IMUDiffusion(n_classes=K, device=dev).fit(xr, yr, steps=steps_diff[0] if reg == "100pct" else steps_diff[1],
                                                      tag=f" {reg}")
        n_syn = max(len(xr), 3000) if full else 300
        ys = np.random.default_rng(2).integers(0, K, n_syn)
        xs = g.sample(ys, seed=3)
        tstr, _ = train_har(xs, ys, xv, yv, dev, steps=steps_clf, seed=0)
        res["fidelity"][reg] = {
            "frechet_distance": frechet_distance(f_test, embed(ref, xs, dev)),
            "frechet_distance_real_train_vs_test": frechet_distance(f_test, embed(ref, xr[:len(xte)], dev)),
            "label_consistency": float((predict_logits(ref, xs, dev).argmax(1) == ys).mean()),
            "tstr_macro_f1": macro_f1(yte, predict_logits(tstr, xte, dev).argmax(1), K),
            "seconds": round(time.time() - t0, 1)}
        fr = res["fidelity"][reg]
        print(f"[behavior] generator {reg}: FD {fr['frechet_distance']:.2f} (real train {fr['frechet_distance_real_train_vs_test']:.2f}) | "
              f"label consistency {fr['label_consistency']:.3f} | TSTR macro-F1 {fr['tstr_macro_f1']:.3f}", flush=True)
        if reg == "100pct":
            np.savez_compressed(od / "synth_har_100pct.npz", x=xs[:600], y=ys[:600])
        configs = {"real": {}, "real+orientation_only_aug": {"augment": rot_only},
                   "real+imu_sensor_model_aug": {"augment": imu},
                   "real+diffusion": {"extra": (xs, ys)},
                   "real+diffusion+imu_sensor_model_aug": {"extra": (xs, ys), "augment": imu}}
        res["results"][reg] = {}
        for name, kw in configs.items():
            runs = []
            for seed in prof.seeds:
                m, _ = train_har(xr, yr, xv, yv, dev, steps=steps_clf, seed=seed, augment=kw.get("augment"),
                                 extra=kw.get("extra"))
                r = {}
                for tn, tx in tests.items():
                    p = predict_logits(m, tx, dev).argmax(1)
                    r[f"{tn}_macro_f1"] = macro_f1(yte, p, K)
                    r[f"{tn}_accuracy"] = float((p == yte).mean())
                runs.append(r)
            summ = {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
                    for k in runs[0]}
            res["results"][reg][name] = {"summary": summ, "runs": runs}
            print(f"[behavior] {reg:>6} | {name:<38} | macro-F1 as recorded {summ['test_as_recorded_macro_f1']['mean']:.3f}"
                  f"±{summ['test_as_recorded_macro_f1']['std']:.3f} | re-oriented phone "
                  f"{summ['test_reoriented_phone_macro_f1']['mean']:.3f}±{summ['test_reoriented_phone_macro_f1']['std']:.3f}",
                  flush=True)
            save_json(res, path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        z = np.load(od / "synth_har_100pct.npz")
        fig = plt.figure(figsize=(15, 7))
        gs = fig.add_gridspec(2, 4)
        t = np.arange(128) / 50
        for k, (cls, col) in enumerate(((0, "walking"), (3, "sitting"))):
            i_r = np.nonzero(yte == cls)[0][5]; i_s = np.nonzero(z["y"] == cls)[0][0]
            for j, (src, xx, ttl) in enumerate(((xte, i_r, "real"), (z["x"], i_s, "generated"))):
                ax = fig.add_subplot(gs[k, j])
                for c, cc in zip(range(3), ("#d6336c", "#2b9e5e", "#1971c2")):
                    ax.plot(t, src[xx, c], color=cc, lw=1)
                ax.set_title(f"{ttl} · {col} · acceleration (g)", fontsize=9); ax.grid(alpha=0.3)
        ax = fig.add_subplot(gs[:, 2:])
        names = list(res["results"]["100pct"])
        for j, (tn, col) in enumerate((("test_as_recorded", "#868e96"), ("test_reoriented_phone", "#d6336c"))):
            v = [res["results"]["100pct"][n]["summary"][f"{tn}_macro_f1"]["mean"] for n in names]
            e = [res["results"]["100pct"][n]["summary"][f"{tn}_macro_f1"]["std"] for n in names]
            ax.barh(np.arange(len(names)) + 0.38 * j, v, 0.38, xerr=e, color=col, label=tn.replace("_", " "))
        ax.set_yticks(np.arange(len(names)) + 0.19, [n.replace("_", " ") for n in names], fontsize=8)
        ax.set_xlabel("macro-F1 on 9 unseen subjects (3 seeds)"); ax.legend(frameon=False, fontsize=8)
        ax.set_xlim(0, 1)
        fig.tight_layout(); fig.savefig(od / "fig_behavior.png", dpi=130)
    except Exception as e:
        print("[behavior] figure skipped:", e)


if __name__ == "__main__":
    main()
