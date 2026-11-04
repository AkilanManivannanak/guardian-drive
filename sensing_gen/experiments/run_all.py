"""Run every sensing_gen experiment in order and write a Markdown summary.

    SG_PROFILE=smoke python -m sensing_gen.experiments.run_all   # ~3 min sanity pass
    SG_PROFILE=full  python -m sensing_gen.experiments.run_all   # all reported numbers
"""
from __future__ import annotations

import json
import sys
import time

import importlib

from .common import get_profile, out_dir

STEPS = ["multiphysics", "calibration", "pinn", "diffusion", "downstream", "sensor_ablation", "guidance", "driving", "fem_crosscheck",
         "physics_guidance", "surrogate", "design", "scaling"]


def fmt(m):
    return f"{m['mean']:.3f} ± {m['std']:.3f}"


def write_report(od):
    L = ["# sensing_gen results", ""]
    p = od / "multiphysics_results.json"
    if p.exists():
        m = json.loads(p.read_text())
        v = m["verification"]
        L += ["## Electrode-skin multiphysics (contact mechanics + EQS finite volume)", "",
              "| electrode | abs(Z) at 10 Hz | kappa (unloading) | kappa (lift-off) | mains diff / V_cm (Zin 10 MOhm) |",
              "|---|---|---|---|---|"]
        for k, e in m["electrodes"].items():
            s = e["sensitivity"]
            L.append(f"| {k} | {e['Z_10Hz_ohm'] / 1e3:.1f} kOhm | {s['kappa_pressure']:.3f} | {s['kappa_liftoff']:.3f} | "
                     f"{e['mains_diff_per_volt_cm_lift20pct']['zin_10MOhm']:.2e} |")
        L += ["", f"Verification: thin-strip analytic error {v['thin_strip_rel_error']:.1e}, reciprocity error "
              f"{v['reciprocity_rel_error']:.1e}, grid-convergence spread "
              f"{max(c['rel_diff_vs_finest'] for c in v['grid_convergence']):.1e}", ""]
    p = od / "calibration_results.json"
    if p.exists():
        c = json.loads(p.read_text())
        L += ["## Simulator calibration (median-beat R²)", "", "| bank | beats | median R² | IQR | R²>0.8 |",
              "|---|---|---|---|---|"]
        for k in ("full", "low10"):
            if k in c:
                v = c[k]
                L.append(f"| {k} | {v['n_beats_fitted']} | {v['r2_median']:.3f} | "
                         f"{v['r2_iqr'][0]:.2f}–{v['r2_iqr'][1]:.2f} | {v['frac_r2_above_0.8']:.2f} |")
        L.append("")
    p = od / "pinn_results.json"
    if p.exists():
        r = json.loads(p.read_text())["real_ptbxl_plus_nstdb"]
        L += ["## Denoising real ECG + real NSTDB noise (output SNR, dB)", "",
              "| input SNR | R-peak F1 | band-pass | NN (no physics) | diff-sim | PINN λ=0.1 | PINN λ=1 |",
              "|---|---|---|---|---|---|---|"]
        for snr, v in r["by_snr"].items():
            s = v["summary"]
            L.append(f"| {snr} dB | {v['rpeak_f1_on_noisy']:.2f} | {fmt(s['bandpass_0.5-40Hz'])} | "
                     f"{fmt(s['nn_no_physics'])} | {fmt(s['diffsim_hard_constraint'])} | "
                     f"{fmt(s['pinn_lam0.1'])} | {fmt(s['pinn_lam1'])} |")
        L.append("")
    p = od / "diffusion_results.json"
    if p.exists():
        d = json.loads(p.read_text())
        L += ["## Generative fidelity vs held-out real test set", "",
              "| source | FED ↓ | MMD² ↓ | LSD dB ↓ | HR W1 ↓ | disc. score ↓ | TSTR AUROC ↑ |",
              "|---|---|---|---|---|---|---|"]
        for k, v in d["fidelity_vs_real_test"].items():
            L.append(f"| {k} | {v['frechet_ecg_distance']:.2f} | {v['mmd2_embedding']:.4f} | "
                     f"{v['log_spectral_distance_db']:.2f} | {v['hr_wasserstein_bpm']:.2f} | "
                     f"{v['discriminative_score']:.3f} | {v['tstr_auroc']:.3f} |")
        L += ["", f"TRTR reference AUROC: {d['trtr_auroc_reference']:.3f}",
              f"HR control MAE: {d['hr_control']['mae_bpm']:.2f} bpm "
              f"(measured {', '.join(f'{x:.1f}' for x in d['hr_control']['measured_mean_bpm'])} for requested 50/70/90/110)", ""]
    p = od / "downstream_results.json"
    if p.exists():
        d = json.loads(p.read_text())["results"]
        L += ["## Downstream AUROC (NORM vs abnormal, mean ± std over seeds)", "",
              "| regime | training data | clean | NSTDB 6 dB | NSTDB 0 dB |", "|---|---|---|---|---|"]
        for reg, cfgs in d.items():
            for k, v in cfgs.items():
                if k == "n_train_real":
                    continue
                a = v["auroc"]
                L.append(f"| {reg} | {k} | {fmt(a['clean'])} | {fmt(a['nstdb_6dB'])} | {fmt(a['nstdb_0dB'])} |")
        L.append("")
    p = od / "sensor_ablation_results.json"
    if p.exists():
        d = json.loads(p.read_text())["results"]
        L += ["## Sensor-model priors: hand-set vs multiphysics-derived (AUROC)", "",
              "| regime | augmentation | clean | NSTDB 6 dB | NSTDB 0 dB |", "|---|---|---|---|---|"]
        for reg, cfgs in d.items():
            for k, v in cfgs.items():
                a = v["auroc"]
                L.append(f"| {reg} | {k} | {fmt(a['clean'])} | {fmt(a['nstdb_6dB'])} | {fmt(a['nstdb_0dB'])} |")
        L.append("")
    p = od / "guidance_results.json"
    if p.exists():
        g = json.loads(p.read_text())
        L += ["## Sampler / guidance sweep (selected on validation)", "",
              "| model | sampler / w | saturated samples | FED (val) | LSD dB |", "|---|---|---|---|---|"]
        for k, sw in g["sweep_on_validation"].items():
            for name, r in sw.items():
                L.append(f"| {k} | {name} | {r['saturated_fraction']:.3f} | {r['frechet_ecg_distance']:.2f} | "
                         f"{r['log_spectral_distance_db']:.2f} |")
        if g.get("final_on_test"):
            L += ["", "| model (selected w) | saturated | FED ↓ | MMD² ↓ | LSD dB ↓ | disc. score ↓ | TSTR AUROC ↑ |",
                  "|---|---|---|---|---|---|---|"]
            for k, r in g["final_on_test"].items():
                L.append(f"| {k} ({r.get('sampler', '')}, w={r['w']}) | {r['saturated_fraction']:.3f} | {r['frechet_ecg_distance']:.2f} | "
                         f"{r['mmd2_embedding']:.4f} | {r['log_spectral_distance_db']:.2f} | "
                         f"{r['discriminative_score']:.3f} | {r['tstr_auroc']:.3f} |")
        if g.get("downstream"):
            L += ["", "| regime | training data | clean | NSTDB 6 dB | NSTDB 0 dB |", "|---|---|---|---|---|"]
            for reg, cfgs in g["downstream"].items():
                for k, v in cfgs.items():
                    a = v["auroc"]
                    L.append(f"| {reg} | {k} | {fmt(a['clean'])} | {fmt(a['nstdb_6dB'])} | {fmt(a['nstdb_0dB'])} |")
        L.append("")
    p = od / "fem_crosscheck.json"
    if p.exists():
        d = json.loads(p.read_text())
        L += ["## Independent finite-element cross-check of the electrode solver", "",
              f"P1 finite elements vs finite volume, wet electrode, 1 Hz–10 kHz: max |Z| difference "
              f"{100 * d['max_rel_error']:.2f} % (mean {100 * d['mean_rel_error']:.2f} %), phase within "
              f"{d['max_phase_diff_deg']:.2f}°; 20 % lift-off raises |Z| {100 * d['liftoff20_rise_fe']:.2f} % (FE) vs "
              f"{100 * d['liftoff20_rise_fv']:.2f} % (FV).", ""]
    p = od / "driving_results.json"
    if p.exists():
        d = json.loads(p.read_text())
        L += ["## Real in-car ECG (PhysioNet drivedb): false cardiac alarms at 90 % validation specificity", "",
              f"{d['n_records']} drives, {d['n_windows']} ten-second windows "
              f"({d['frac_windows_detectable_beats']:.2f} with detectable beats).", "",
              "| training | PTB-XL NORM false alarms | in-car, all | in-car, first 10 min | in-car, driving | PTB-XL AUROC |",
              "|---|---|---|---|---|---|"]
        for k, v in d["results"].items():
            s_ = v["summary"]
            rest = fmt(s_["drive_false_alarm_rest"]) if "drive_false_alarm_rest" in s_ else "–"
            L.append(f"| {k} | {fmt(s_['ptbxl_test_norm_false_alarm'])} | {fmt(s_['drive_false_alarm_all'])} | "
                     f"{rest} | {fmt(s_['drive_false_alarm_driving'])} | {fmt(s_['ptbxl_test_auroc'])} |")
        for k, v in d.get("bootstrap_vs_real", {}).items():
            L.append(f"\n{k}: change in in-car false alarms vs real-only {v['delta_false_alarm']:+.3f} "
                     f"(95 % driver-bootstrap CI {v['ci95'][0]:+.3f} to {v['ci95'][1]:+.3f})")
        L.append("")
    p = od / "physics_guidance_results.json"
    if p.exists():
        d = json.loads(p.read_text())
        L += ["## Physics-informed (limit-cycle) guidance for the diffusion sampler (no retraining)", "",
              f"Selected on validation: λ = {d.get('selected_lam', '–')}", "",
              "| sampler | HR-control MAE (bpm) ↓ | FED ↓ | MMD² ↓ | LSD dB ↓ | disc. ↓ | TSTR AUROC ↑ |",
              "|---|---|---|---|---|---|---|"]
        for k, r in d.get("final_on_test", {}).items():
            L.append(f"| {k} (λ={r['lam']:g}) | {r['hr_control']['mae_bpm']:.2f} | {r['frechet_ecg_distance']:.2f} | "
                     f"{r['mmd2_embedding']:.4f} | {r['log_spectral_distance_db']:.2f} | "
                     f"{r['discriminative_score']:.3f} | {r['tstr_auroc']:.3f} |")
        L.append("")
    p = od / "surrogate_results.json"
    if p.exists():
        d = json.loads(p.read_text())
        L += ["## Neural surrogate of the electrode simulator (held-out FE spectra)", "",
              "| model | median abs(Z) error | p95 abs(Z) error | median phase error | 95 % coverage |", "|---|---|---|---|---|"]
        for k, r in d["models"].items():
            cov = f"{r['coverage_95']:.2f}" if "coverage_95" in r else "–"
            L.append(f"| {k} | {100 * r['absZ_rel_err_median']:.2f} % | {100 * r['absZ_rel_err_p95']:.2f} % | "
                     f"{r['phase_err_deg_median']:.3f}° | {cov} |")
        if "speed" in d:
            L += ["", f"Speed: {d['speed']['speedup']:,.0f}x faster than the FE solver per spectrum."]
        L.append("")
    p = od / "design_results.json"
    if p.exists():
        d = json.loads(p.read_text())
        L += ["## Robust electrode design through the differentiable surrogate (verified by FE)", "",
              "| design | w (mm) | s (mm) | FE objective J | SNR vs max-spacing | regret vs FE grid |", "|---|---|---|---|---|---|"]
        for k, r in d["designs"].items():
            L.append(f"| {k} | {r['w_mm']:.2f} | {r['s_mm']:.2f} | {r['J_fe_mean']:.4f} | "
                     f"{r['snr_gain_db_vs_max_spacing']:+.2f} dB | {r['regret_vs_fe_grid']:.4f} |")
        L += ["", f"Surrogate optimisation {d['surrogate_opt_seconds']:.1f} s vs FE grid search {d['fe_grid_seconds']:.0f} s.", ""]
    p = od / "scaling_results.json"
    if p.exists() and json.loads(p.read_text()).get("synthetic"):
        d = json.loads(p.read_text())
        L += ["## How much real data does synthetic data replace? (sensor-model augmentation in all runs)", "",
              "| training data | clean AUROC | ≈ real data | NSTDB 0 dB AUROC | ≈ real data |", "|---|---|---|---|---|"]
        for f in d["fractions"]:
            r = d["real_only"][f"{f:g}"]
            L.append(f"| {100 * f:.0f} % real only | {fmt(r['clean'])} | – | {fmt(r['nstdb_0dB'])} | – |")
        for name, rows in d["synthetic"].items():
            for n, r in rows.items():
                e = r["equivalent_real_fraction"]
                L.append(f"| 10 % real + {n} {name} | {fmt(r['clean'])} | {100 * e['clean']:.0f} % | "
                         f"{fmt(r['nstdb_0dB'])} | {100 * e['nstdb_0dB']:.0f} % |")
        L.append("")
    (od / "RESULTS.md").write_text("\n".join(L))
    print("\n".join(L))


def main():
    prof = get_profile()
    od = out_dir(prof)
    only = sys.argv[1:] or STEPS
    for name in STEPS:
        if name not in only:
            continue
        mod = importlib.import_module(f"sensing_gen.experiments.run_{name}")
        t0 = time.time()
        print(f"\n===== [{prof.name}] {name} =====", flush=True)
        mod.main()
        print(f"===== {name} done in {time.time() - t0:.0f} s =====", flush=True)
    write_report(od)


if __name__ == "__main__":
    main()
