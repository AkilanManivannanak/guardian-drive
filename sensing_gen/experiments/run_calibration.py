"""Step 1: calibrate the ECG dynamical model to real PTB-XL median beats (differentiable simulation)."""
from __future__ import annotations

import time

import numpy as np

from ..data.signal_utils import median_beat
from ..physics.calibrate import build_bank, fit_beats
from .common import env_info, get_profile, load_splits, low_data_indices, out_dir, save_json


def main():
    prof = get_profile()
    od = out_dir(prof)
    ds, train, val, test = load_splits(prof)
    print(f"[calib] PTB-XL single-lead: train {len(train)}  val {len(val)}  test {len(test)}  "
          f"abnormal rate {train.y.mean():.3f}  scale {ds.scale:.4f} mV/unit")
    low = low_data_indices(train.y, 0.10)
    np.save(od / "low10_indices.npy", low)

    res = {"env": env_info(), "n_train": len(train), "n_val": len(val), "n_test": len(test),
           "abnormal_rate_train": float(train.y.mean()), "mv_per_unit": ds.scale}
    for name, sub in (("full", train), ("low10", train.subset(low))):
        t0 = time.time()
        bank = build_bank(sub.x, sub.y, sub.hr, sub.rr_std, steps=prof.calib_steps, verbose=True)
        bank.save(od / f"param_bank_{name}.npz")
        r2 = bank.r2
        res[name] = {
            "n_records": len(sub), "n_beats_fitted": int(len(r2)),
            "r2_median": float(np.median(r2)), "r2_iqr": [float(np.percentile(r2, 25)), float(np.percentile(r2, 75))],
            "frac_r2_above_0.8": float(np.mean(r2 > 0.8)),
            "r2_median_by_class": {("NORM" if c == 0 else "ABN"): float(np.median(r2[bank.label == c]))
                                   for c in np.unique(bank.label)},
            "seconds": round(time.time() - t0, 1),
        }
        print(f"[calib] {name}: {len(r2)} beats, median R^2 {np.median(r2):.3f}")

    # Figure: real vs calibrated median beats
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        beats, rrs, labs = [], [], []
        rng = np.random.default_rng(1)
        for c in (0, 1):
            for i in rng.choice(np.nonzero(test.y == c)[0], 12, replace=False):
                b, rr = median_beat(test.x[i])
                if b is not None and len(beats) < (c + 1) * 4:
                    beats.append(b); rrs.append(rr); labs.append(c)
        _, r2, fitted = fit_beats(np.stack(beats), np.array(rrs), steps=prof.calib_steps)
        fig, ax = plt.subplots(2, 4, figsize=(13, 5.5), sharex=True)
        t = (np.arange(beats[0].size) - 30) / 100.0
        for k, a in enumerate(ax.flat[: len(beats)]):
            a.plot(t, beats[k], color="#222", lw=1.6, label="real median beat")
            a.plot(t, fitted[k], color="#d6336c", lw=1.4, ls="--", label="calibrated ODE")
            a.set_title(f"{'NORM' if labs[k] == 0 else 'ABNORMAL'}  R²={r2[k]:.2f}", fontsize=10)
            a.grid(alpha=0.3)
        ax[0, 0].legend(fontsize=8)
        fig.supxlabel("time from R peak [s]"); fig.supylabel("lead II [normalised]")
        fig.suptitle("McSharry ECG ODE calibrated to held-out PTB-XL beats by differentiable simulation")
        fig.tight_layout()
        fig.savefig(od / "fig_calibration.png", dpi=130)
    except Exception as e:  # plotting is optional
        print("[calib] figure skipped:", e)

    save_json(res, od / "calibration_results.json")


if __name__ == "__main__":
    main()
