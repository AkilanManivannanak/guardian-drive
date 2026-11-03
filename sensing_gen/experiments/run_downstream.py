"""Step 4: does simulation / generation make a wearable-ECG classifier better and more robust?

Task: NORM vs abnormal (MI/STTC/CD/HYP) from a single 10-s lead, PTB-XL folds 1-8 / 9 / 10.
Test conditions: clean, and corrupted with *real* recorded sensor noise (MIT-BIH NSTDB, held-out
half of every recording) at 6 dB and 0 dB. No configuration ever sees the test-half noise.
"""
from __future__ import annotations

import time

import numpy as np

from ..data.datasets import NoiseBank, corrupt_at_snr
from ..eval.models import auroc, predict, train_classifier
from ..physics.sensor import ElectrodeSkinSensor
from .common import DATA, device, env_info, get_profile, load_splits, out_dir, save_json


def main():
    prof = get_profile()
    od = out_dir(prof)
    dev = device()
    _, train, val, test = load_splits(prof)
    low = np.load(od / "low10_indices.npy")
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

    test_sets = {"clean": test.x}
    for snr in (6, 0):
        noise = nb.sample(len(test), test.x.shape[1], "test", np.random.default_rng(1000 + snr))
        test_sets[f"nstdb_{snr}dB"] = corrupt_at_snr(test.x, noise, snr)

    def synth(name):
        z = np.load(od / f"synth_{name}.npz")
        return z["x"], z["y"]

    regimes = {
        "10pct_real": (train.subset(low), {
            "real": {},
            "real+sensor_model_aug": {"augment": sensor},
            "real+nstdb_noise_aug(reference)": {"augment": nstdb_aug},
            "real+simulator": {"extra": synth("sim_low10")},
            "real+diffusion": {"extra": synth("diff_low10")},
            "real+sim_pretrained_diffusion": {"extra": synth("diff_simpre_low10")},
            "real+sim_pretrained_diffusion+sensor_model_aug": {"extra": synth("diff_simpre_low10"),
                                                              "augment": sensor},
        }),
        "100pct_real": (train, {
            "real": {},
            "real+sensor_model_aug": {"augment": sensor},
            "real+nstdb_noise_aug(reference)": {"augment": nstdb_aug},
            "real+simulator": {"extra": synth("sim_full")},
            "real+diffusion": {"extra": synth("diff_full")},
            "real+diffusion+sensor_model_aug": {"extra": synth("diff_full"), "augment": sensor},
        }),
    }
    path = od / "downstream_results.json"
    res = {"env": env_info(), "n_test": len(test), "results": {}}
    for regime, (tr, configs) in regimes.items():
        res["results"][regime] = {"n_train_real": len(tr)}
        for cfg_name, kw in configs.items():
            runs = []
            t0 = time.time()
            for seed in prof.seeds:
                m, info = train_classifier(tr.x, tr.y, val.x, val.y, dev, steps=prof.clf_steps,
                                           seed=seed, extra=kw.get("extra"), augment=kw.get("augment"),
                                           p_extra=0.5)
                runs.append({k: auroc(test.y, predict(m, v, dev)) for k, v in test_sets.items()})
            summ = {k: {"mean": float(np.mean([r[k] for r in runs])),
                        "std": float(np.std([r[k] for r in runs]))} for k in test_sets}
            res["results"][regime][cfg_name] = {"auroc": summ, "runs": runs,
                                                "seconds": round(time.time() - t0, 1)}
            print(f"[downstream] {regime:>11} | {cfg_name:<50} | " +
                  " | ".join(f"{k} {v['mean']:.3f}±{v['std']:.3f}" for k, v in summ.items()), flush=True)
            save_json(res, path)


if __name__ == "__main__":
    main()
