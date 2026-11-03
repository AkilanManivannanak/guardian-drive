"""Step 5: close the loop multiphysics -> sensor model -> ML training.

Same downstream task as run_downstream, comparing sensor-model augmentation with the original
hand-set priors vs priors derived from the electrode-skin field model (run_multiphysics).
"""
from __future__ import annotations

import time

import numpy as np

from ..data.datasets import NoiseBank, corrupt_at_snr
from ..eval.models import auroc, predict, train_classifier
from ..physics.sensor import ElectrodeSkinSensor, SensorConfig
from .common import DATA, device, env_info, get_profile, load_splits, out_dir, save_json


def main():
    prof = get_profile()
    od = out_dir(prof)
    dev = device()
    _, train, val, test = load_splits(prof)
    low = np.load(od / "low10_indices.npy")
    nb = NoiseBank(DATA / "nstdb")
    mp_cfg = SensorConfig.from_multiphysics(od / "multiphysics_results.json")
    sensors = {"sensor_model_aug(hand-set priors)": ElectrodeSkinSensor(SensorConfig()),
               "sensor_model_aug(multiphysics priors)": ElectrodeSkinSensor(mp_cfg)}
    test_sets = {"clean": test.x}
    for snr in (6, 0):
        noise = nb.sample(len(test), test.x.shape[1], "test", np.random.default_rng(1000 + snr))
        test_sets[f"nstdb_{snr}dB"] = corrupt_at_snr(test.x, noise, snr)
    res = {"env": env_info(), "multiphysics_priors": {"kappa_range": mp_cfg.kappa_range,
                                                      "ze_ratio_range": mp_cfg.ze_ratio_range},
           "results": {}}
    for regime, tr in (("10pct_real", train.subset(low)), ("100pct_real", train)):
        res["results"][regime] = {}
        for name, sensor in sensors.items():
            t0, runs = time.time(), []
            for seed in prof.seeds:
                m, _ = train_classifier(tr.x, tr.y, val.x, val.y, dev, steps=prof.clf_steps, seed=seed,
                                        augment=sensor)
                runs.append({k: auroc(test.y, predict(m, v, dev)) for k, v in test_sets.items()})
            summ = {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
                    for k in test_sets}
            res["results"][regime][name] = {"auroc": summ, "runs": runs, "seconds": round(time.time() - t0, 1)}
            print(f"[sensor-ablation] {regime:>11} | {name:<40} | " +
                  " | ".join(f"{k} {v['mean']:.3f}±{v['std']:.3f}" for k, v in summ.items()), flush=True)
            save_json(res, od / "sensor_ablation_results.json")


if __name__ == "__main__":
    main()
