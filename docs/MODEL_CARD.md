# Model card: sensing_gen library

Three components you can plug into a training or evaluation pipeline. Install with
`pip install -e ".[sensing]"` from the repository root; the CLI is `sensing-gen`.

```python
from sensing_gen import api
noisy = api.SensorAugmenter(seed=0)(ecg_mv)                    # (N, L) mV, 100 Hz
x_mv, y = api.ECGGenerator.from_pretrained().sample(64, "abnormal", hr_bpm=80)
z, sigma = api.ElectrodeSurrogate.from_pretrained().impedance(w_el_mm=10, lift=0.2)
```

```bash
sensing-gen augment ecg.npy ecg_noisy.npy --fs 250 --copies 4
sensing-gen generate synth.npz --n 500 --label norm --hr 65
sensing-gen impedance --w-el-mm 8 --lift 0.3          # add --exact for the FE solver
```

## 1. Physics sensor model (`SensorAugmenter`)

| | |
|---|---|
| What it does | Corrupts clean single-lead ECG with physically parameterised wearable artifacts: seat/torso motion → skin-stretch potential, motion-modulated electrode impedance (gain change), impedance-mismatch mains pickup (60 Hz aliased to 40 Hz at 100 Hz), burst EMG, respiration baseline, contact loss, 12-bit ADC. |
| Trained on | Nothing (no learned parameters). Priors are hand-set; a variant takes priors from the multiphysics model (`multiphysics_priors=`). |
| Evidence | As training augmentation it raised AUROC under real, held-out MIT-BIH NSTDB noise at 0 dB from 0.654 to 0.818 (PTB-XL, 3 seeds), above augmentation with the real recorded noise (0.807). |
| Not shown | It did not reduce false alarms on real in-car ECG (PhysioNet drivedb), which was clean. It has not been fitted to a specific device; `run_wearable.py` is the check for that. |

## 2. ECG generator (`ECGGenerator`)

| | |
|---|---|
| Model | Conditional 1-D U-Net DDPM (FiLM on class and heart rate), DDIM sampler with leading timesteps and dynamic thresholding, classifier-free guidance w = 1. 3.4 M parameters, `weights/ecg_diffusion_ptbxl_lead2.pt`. |
| Trained on | PTB-XL (CC BY 4.0) lead II, 100 Hz, 10-s windows, folds 1–8 (3,960 records); labels NORM vs abnormal (MI / STTC / CD / HYP). |
| Output | 10-s ECG in mV; heart rate controllable to about 4 bpm (MAE). |
| Evidence (test fold 10) | Fréchet ECG distance 1.28, real-vs-fake discriminator accuracy 0.63 (0.5 ideal); a classifier trained only on generated ECG: AUROC 0.829 vs 0.801 for real data. 10 % real + 8,000 generated records ≈ 100 % real (0.822 vs 0.824, 3 seeds). |
| Limits | One lead, resting clinical ECG, two coarse classes. Not a diagnostic device and not validated for rare arrhythmias. Do not use generated ECG to evaluate a model (only to train it). |

## 3. Electrode surrogate (`ElectrodeSurrogate`)

| | |
|---|---|
| Model | 5-member deep ensemble, physics-informed: learns the correction to an analytic contact-impedance prior. `weights/electrode_surrogate.pt`. |
| Trained on | 4,000 finite-element spectra (21 frequencies, 1 Hz–10 kHz) over the ranges in `surrogate/space.py`: dermis 1–2.5 mm, fat 1–10 mm, stratum-corneum conductivity 5e-6–1e-4 S/m, electrode width 5–15 mm, spacing 20–40 mm, lift-off 0–40 %, and tissue conductivities. |
| Evidence | Median \|Z\| error 0.02 % on 1,000 held-out FE spectra (at the FE mesh-discretisation level), 95 % intervals cover 98 %, 37,714× faster than the solver. |
| Limits | 2-D electro-quasi-static model, wet electrode, nominal literature tissue properties; outside the training ranges the error grows (0.16 % at +25 %), which the returned sigma flags. |

## Reproducing the weights

`SG_PROFILE=full python -m sensing_gen.experiments.run_all diffusion surrogate` writes `outputs/sensing_gen/ckpt_diff_full.pt` and `ckpt_surrogate.pt`; the shipped weights are copies of those files.
