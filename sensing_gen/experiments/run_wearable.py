"""Step 14: does the physics sensor model match a *real wearable* I recorded myself?

Data: ECG recorded by the author on a consumer device (Apple Watch / Polar H10 / AD8232; see
data/wearable_io.py and docs/wearable_protocol.md), one folder per condition under
data/sensing_gen/wearable/: a reference condition (still / rest / sitting) and motion conditions
(walking, arm motion, car passenger, ...). The raw files are never committed: they contain personal
health data. Only aggregate results are written to outputs/.

Questions
 1. Signal quality: how much do motion conditions degrade beat detectability and beat-to-beat
    consistency (template-correlation SQI) on this device?
 2. Sim-to-real for the noise model: the motion-artifact spectrum is estimated as the PSD excess of
    each motion condition over the still condition. Its *shape* is compared with (a) the physics
    sensor model's artifact (sensor(x) - x applied to the still windows) and (b) real NSTDB noise
    (training half). Lower log-spectral distance = better match.
 3. Deployment: false cardiac-alarm rate per condition for classifiers trained on PTB-XL with and
    without sensor-model augmentation (threshold = 90 % specificity on PTB-XL validation NORM;
    nothing tuned on wearable data). The recorder is assumed to have no known cardiac condition, so
    every "abnormal" output is a false alarm. Recording-level bootstrap CIs.
Amplitude: one gain per device, set so that the *still* windows match the PTB-XL training median
window std; motion windows keep their extra noise power.
Caveat: wrist devices record lead I, PTB-XL models were trained on lead II.
"""
from __future__ import annotations

import numpy as np

from ..data.datasets import NoiseBank
from ..data.signal_utils import detect_rpeaks
from ..data.wearable_io import load_folder
from ..eval.models import predict, train_classifier
from ..eval.wearable import beat_sqi, excess_psd, is_reference, shape_match, to_windows, band_psd
from ..physics.sensor import ElectrodeSkinSensor
from .common import DATA, device, env_info, get_profile, load_splits, out_dir, save_json


def main():
    import torch
    prof = get_profile()
    od = out_dir(prof)
    dev = device()
    root = DATA / "wearable"
    if not root.exists():
        print(f"[wearable] skipped: no recordings in {root} (see docs/wearable_protocol.md)")
        return
    recs = load_folder(root)
    if not recs:
        print(f"[wearable] skipped: no readable recordings in {root}")
        return
    _, train, val, test = load_splits(prof)
    ref_std = float(np.median(train.x.std(1)))

    W, cond_of, rec_of, names = [], [], [], []
    for i, r in enumerate(recs):
        w = to_windows(r.x, r.fs)
        if len(w) == 0:
            print(f"[wearable] {r.condition}/{r.name}: shorter than 10 s, skipped")
            continue
        W.append(w); cond_of += [r.condition] * len(w); rec_of += [i] * len(w)
        print(f"[wearable] {r.condition}/{r.name}: {r.fmt}, {r.fs:g} Hz, {len(r.x) / r.fs:.0f} s -> {len(w)} windows", flush=True)
    X = np.concatenate(W)
    cond_of, rec_of = np.array(cond_of), np.array(rec_of)
    conds = sorted(set(cond_of), key=lambda c: (not is_reference(c), c))
    ref = [c for c in conds if is_reference(c)]
    gain_src = np.isin(cond_of, ref) if ref else np.ones(len(X), bool)
    gain = ref_std / (np.median(X[gain_src].std(1)) + 1e-8)
    X = np.clip(X * gain, -12, 12).astype(np.float32)
    res = {"env": env_info(), "n_recordings": len(recs), "n_windows": int(len(X)), "reference_conditions": ref,
           "devices": sorted({r.fmt for r in recs}), "leads": sorted({r.lead for r in recs if r.lead}),
           "conditions": {}, "artifact_spectrum": {}, "false_alarms": {}}

    # 1) signal quality per condition
    for c in conds:
        m = cond_of == c
        nb = np.array([len(detect_rpeaks(w)) for w in X[m]])
        sqi = np.array([beat_sqi(w) for w in X[m]])
        res["conditions"][c] = {"n_recordings": int(len(set(rec_of[m]))), "n_windows": int(m.sum()),
                                "frac_detectable_beats": float(((nb >= 4) & (nb <= 30)).mean()),
                                "beat_sqi_median": float(np.nanmedian(sqi)) if np.isfinite(sqi).any() else None,
                                "window_std_rel_to_still": float(np.median(X[m].std(1)) / np.median(X[gain_src].std(1)))}
        q = res["conditions"][c]
        print(f"[wearable] {c:<16} recs {q['n_recordings']:>2} windows {q['n_windows']:>4} | detectable beats "
              f"{q['frac_detectable_beats']:.2f} | beat SQI {q['beat_sqi_median'] if q['beat_sqi_median'] is None else round(q['beat_sqi_median'], 3)} "
              f"| std x{q['window_std_rel_to_still']:.2f}", flush=True)

    # 2) artifact spectrum: real excess vs sensor model vs NSTDB
    if ref:
        still = X[np.isin(cond_of, ref)]
        sensor = ElectrodeSkinSensor()
        g = torch.Generator().manual_seed(0)
        xs = torch.tensor(still)
        sim_art = (sensor(xs, g) - xs).numpy()
        _, p_sim = band_psd(sim_art)
        nb = NoiseBank(DATA / "nstdb")
        _, p_nst = band_psd(nb.sample(len(still), still.shape[1], "train", np.random.default_rng(0)))
        for c in conds:
            if c in ref:
                continue
            f, ex, snr = excess_psd(X[cond_of == c], still)
            res["artifact_spectrum"][c] = {"artifact_snr_db": snr, "vs_sensor_model": shape_match(ex, p_sim),
                                           "vs_nstdb": shape_match(ex, p_nst), "freq_hz": f, "excess_psd": ex}
            a = res["artifact_spectrum"][c]
            print(f"[wearable] artifact {c:<16} SNR {snr:5.1f} dB | shape LSD vs sensor model {a['vs_sensor_model']['lsd_db']:.2f} dB "
                  f"(r={a['vs_sensor_model']['log_psd_corr']:.2f}) | vs NSTDB {a['vs_nstdb']['lsd_db']:.2f} dB "
                  f"(r={a['vs_nstdb']['log_psd_corr']:.2f})", flush=True)
        res["sensor_model_psd"], res["nstdb_psd"] = p_sim, p_nst

    # 3) false alarms per condition
    val_norm = val.x[val.y == 0]
    rng = np.random.default_rng(0)
    configs = {"real": {}, "real+sensor_model_aug": {"augment": ElectrodeSkinSensor()}}
    flags = {}
    for name, kw in configs.items():
        per_seed = []
        for seed in prof.seeds:
            m, _ = train_classifier(train.x, train.y, val.x, val.y, dev, steps=prof.clf_steps, seed=seed,
                                    augment=kw.get("augment"))
            thr = float(np.quantile(predict(m, val_norm, dev), 0.90))
            per_seed.append(predict(m, X, dev) > thr)
        fl = np.mean(per_seed, 0)
        flags[name] = fl
        res["false_alarms"][name] = {}
        for c in conds:
            mm = cond_of == c
            recs_c = np.unique(rec_of[mm])
            boot = []
            for _ in range(2000):
                pick = rng.choice(recs_c, len(recs_c))
                boot.append(np.concatenate([fl[(rec_of == i)] for i in pick]).mean())
            res["false_alarms"][name][c] = {"rate": float(fl[mm].mean()),
                                            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
        print(f"[wearable] {name:<24} false alarms: " + " | ".join(
            f"{c} {100 * v['rate']:.1f} % [{100 * v['ci95'][0]:.0f}-{100 * v['ci95'][1]:.0f}]"
            for c, v in res["false_alarms"][name].items()), flush=True)
    save_json(res, od / "wearable_results.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 3.8), gridspec_kw={"width_ratios": [1.6, 1, 1]})
        t = np.arange(1000) / 100
        for k, c in enumerate(conds[:3]):
            i = np.nonzero(cond_of == c)[0]
            ax[0].plot(t, X[i[len(i) // 2]] - 8 * k, lw=0.7, label=c)
        ax[0].set_yticks([]); ax[0].set_xlabel("time [s]"); ax[0].legend(frameon=False, fontsize=8)
        ax[0].set_title("My own wearable ECG (random windows, offset)", fontsize=10)
        if res["artifact_spectrum"]:
            c0 = next(iter(res["artifact_spectrum"]))
            a = res["artifact_spectrum"][c0]
            f = a["freq_hz"]
            for p, lab, col in ((a["excess_psd"], f"real motion artifact ({c0})", "#222"),
                                (res["sensor_model_psd"], "physics sensor model", "#d6336c"),
                                (res["nstdb_psd"], "NSTDB recorded noise", "#1971c2")):
                ax[1].semilogy(f, p / p.sum(), color=col, lw=1.5, label=lab)
            ax[1].set_xlabel("frequency [Hz]"); ax[1].set_ylabel("normalised PSD"); ax[1].legend(frameon=False, fontsize=7)
            ax[1].set_title("Artifact spectrum shape", fontsize=10)
        for j, (name, col) in enumerate((("real", "#868e96"), ("real+sensor_model_aug", "#d6336c"))):
            v = [res["false_alarms"][name][c]["rate"] for c in conds]
            ax[2].bar(np.arange(len(conds)) + 0.38 * j, v, 0.38, color=col, label=name)
        ax[2].axhline(0.10, color="k", ls="--", lw=0.8)
        ax[2].set_xticks(np.arange(len(conds)) + 0.19, conds, fontsize=8, rotation=20)
        ax[2].set_ylabel("false cardiac alarms"); ax[2].legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(od / "fig_wearable.png", dpi=130)
    except Exception as e:
        print("[wearable] figure skipped:", e)


if __name__ == "__main__":
    main()
