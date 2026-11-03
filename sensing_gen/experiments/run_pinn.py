"""Step 2: inverse PINN on real ECG corrupted by real sensor noise + parameter recovery on simulation."""
from __future__ import annotations

import time

import numpy as np
import torch

from ..data.datasets import NoiseBank, corrupt_at_snr
from ..data.signal_utils import bandpass, detect_rpeaks, snr_db
from ..physics.ecg_ode import DEFAULT_A, DEFAULT_B, DEFAULT_THETA, ECGParams, WAVES, r_wave_gain
from ..physics.sensor import ElectrodeSkinSensor, SensorConfig
from ..pinn.ecg_pinn import fit_diffsim, fit_pinn, phase_from_rpeaks, simulate_from_phase
from .common import DATA, device, env_info, get_profile, load_splits, out_dir, save_json

FS = 100.0


def peak_f1(ref: np.ndarray, est: np.ndarray, tol: int = 5) -> float:
    if len(ref) == 0 or len(est) == 0:
        return 0.0
    used, tp = set(), 0
    for r in ref:
        j = np.argmin(np.abs(est - r))
        if abs(est[j] - r) <= tol and j not in used:
            used.add(j); tp += 1
    return 2 * tp / (len(ref) + len(est))


def score(clean, est):
    """Output SNR after the same 0.5-40 Hz acquisition band-pass is applied to reference and estimate."""
    return [snr_db(bandpass(c, FS), bandpass(e, FS)) for c, e in zip(clean, est)]


def real_experiment(prof, dev, test):
    nb = NoiseBank(DATA / "nstdb")
    rng = np.random.default_rng(0)
    cand = [i for i in rng.permutation(len(test)) if len(detect_rpeaks(test.x[i])) >= 6]
    sel = np.array(cand[: prof.pinn_records])
    clean = test.x[sel]
    out = {"n_records": int(len(sel)), "abnormal_fraction": float(test.y[sel].mean()), "by_snr": {}}
    for snr in (0, 6, 12):
        noise = nb.sample(len(sel), clean.shape[1], "test", np.random.default_rng(100 + snr))
        noisy = corrupt_at_snr(clean, noise, snr)
        rp_ref = [detect_rpeaks(c) for c in clean]
        rp_idx = [detect_rpeaks(bandpass(n, FS)) for n in noisy]
        f1 = float(np.mean([peak_f1(a, b) for a, b in zip(rp_ref, rp_idx)]))
        rp = [r / FS for r in rp_idx]
        t0 = time.time()
        res = {"input_snr_db": snr, "rpeak_f1_on_noisy": f1}
        res["bandpass_0.5-40Hz"] = score(clean, noisy)
        nn_only = fit_pinn(noisy, rp, steps=prof.pinn_steps, lam=0.0, init_from_diffsim=False, device=dev)
        res["nn_no_physics"] = score(clean, nn_only.denoised)
        dsim = fit_diffsim(noisy, rp, steps=max(50, prof.pinn_steps // 2), device="cpu")
        res["diffsim_hard_constraint"] = score(clean, dsim.denoised)
        for lam in (0.1, 1.0):
            pinn = fit_pinn(noisy, rp, steps=prof.pinn_steps, lam=lam, device=dev,
                            diffsim_steps=max(50, prof.pinn_steps // 4))
            res[f"pinn_lam{lam:g}"] = score(clean, pinn.denoised)
        summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                   for k, v in res.items() if isinstance(v, list)}
        res["summary"] = summary
        res["seconds"] = round(time.time() - t0, 1)
        out["by_snr"][str(snr)] = res
        print(f"[pinn] input {snr:>2} dB | R-peak F1 {f1:.2f} | " +
              " | ".join(f"{k} {v['mean']:.2f}" for k, v in summary.items()), flush=True)
        if snr == 6:
            example = {"clean": clean[0], "noisy": noisy[0], "bandpass": bandpass(noisy[0], FS),
                       "pinn": bandpass(pinn.denoised[0], FS), "nn": bandpass(nn_only.denoised[0], FS)}
    return out, example


def synthetic_recovery(prof, dev):
    """Known ground-truth parameters -> simulate -> sensor noise (6 dB) -> recover."""
    S = prof.pinn_records
    rng = np.random.default_rng(7)
    g = r_wave_gain()
    th = np.array(DEFAULT_THETA)[None] + rng.normal(0, 0.05, (S, 5))
    a = np.array(DEFAULT_A)[None] * g * rng.uniform(0.6, 1.6, (S, 1)) * rng.uniform(0.75, 1.25, (S, 5))
    b = np.array(DEFAULT_B)[None] * rng.uniform(0.8, 1.25, (S, 5))
    p_true = ECGParams(torch.tensor(th, dtype=torch.float32), torch.tensor(a, dtype=torch.float32),
                       torch.tensor(b, dtype=torch.float32), torch.zeros(S))
    rpeaks = []
    for i in range(S):
        mean_rr = 60.0 / rng.uniform(55, 100)
        rr = np.clip(rng.normal(mean_rr, 0.03, 40), 0.35, 1.6)
        rpeaks.append(np.cumsum(rr) - rng.uniform(0, mean_rr) - 3.0)
    fns = [lambda tt, r=r: phase_from_rpeaks(r, tt) for r in rpeaks]
    with torch.no_grad():
        z, _ = simulate_from_phase(p_true, fns, 10.0)
    clean = z[:, ::5][:, :1000].numpy()  # raw ODE output (fit target); scored after band-pass
    sensor = ElectrodeSkinSensor(SensorConfig(p_contact_loss=0.0))
    noisy = sensor(torch.tensor(clean), torch.Generator().manual_seed(0),
                   snr_db=torch.full((S,), 6.0)).numpy()
    rp = [r[(r >= 0) & (r < 10)] for r in rpeaks]
    out = {"n": S, "input_snr_db": 6.0}
    for name, fit in (("diffsim", lambda: fit_diffsim(noisy, rp, steps=max(50, prof.pinn_steps // 2))),
                      ("pinn", lambda: fit_pinn(noisy, rp, steps=prof.pinn_steps, lam=1.0, device=dev,
                                                diffsim_steps=max(50, prof.pinn_steps // 4)))):
        r = fit()
        pe = r.params.to("cpu")
        err = {}
        for k, w in enumerate(WAVES):
            err[w] = {
                "theta_abs_err_rad": float(np.median(np.abs(pe.theta[:, k].numpy() - th[:, k]))),
                "a_rel_err": float(np.median(np.abs(pe.a[:, k].numpy() / a[:, k] - 1))),
                "b_rel_err": float(np.median(np.abs(pe.b[:, k].numpy() / b[:, k] - 1))),
                # a and b trade off; the event strength a*b^2 (area of the wave) is identifiable
                "strength_ab2_rel_err": float(np.median(np.abs(
                    (pe.a[:, k].numpy() * pe.b[:, k].numpy() ** 2) / (a[:, k] * b[:, k] ** 2) - 1))),
            }
        out[name] = {"param_error_median": err,
                     "output_snr_db": float(np.mean(score(clean, r.denoised)))}
        print(f"[pinn] synthetic {name}: R strength rel err {err['R']['strength_ab2_rel_err']:.3f}, "
              f"T angle err {err['T']['theta_abs_err_rad']:.3f} rad, "
              f"SNR out {out[name]['output_snr_db']:.2f} dB", flush=True)
    out["bandpass_output_snr_db"] = float(np.mean(score(clean, noisy)))
    return out


def main():
    prof = get_profile()
    od = out_dir(prof)
    dev = device()
    _, _, _, test = load_splits(prof)
    real, ex = real_experiment(prof, dev, test)
    syn = synthetic_recovery(prof, dev)
    save_json({"env": env_info(), "real_ptbxl_plus_nstdb": real, "synthetic_recovery": syn},
              od / "pinn_results.json")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.arange(len(ex["clean"])) / FS
        fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        ax[0].plot(t, ex["noisy"], color="#999", lw=0.8, label="noisy input (real NSTDB noise, 6 dB)")
        ax[0].plot(t, ex["clean"], color="#222", lw=1.0, label="clean reference")
        ax[1].plot(t, ex["bandpass"], color="#1971c2", lw=0.9, label="band-pass 0.5-40 Hz")
        ax[1].plot(t, ex["clean"], color="#222", lw=0.8, alpha=0.5)
        ax[2].plot(t, ex["pinn"], color="#d6336c", lw=0.9, label="inverse PINN")
        ax[2].plot(t, ex["clean"], color="#222", lw=0.8, alpha=0.5)
        for a_ in ax:
            a_.legend(loc="upper right", fontsize=8); a_.grid(alpha=0.3)
        ax[-1].set_xlabel("time [s]")
        fig.tight_layout()
        fig.savefig(od / "fig_pinn_denoising.png", dpi=130)
    except Exception as e:
        print("[pinn] figure skipped:", e)


if __name__ == "__main__":
    main()
