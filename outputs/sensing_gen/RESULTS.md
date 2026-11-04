# sensing_gen results

## Electrode-skin multiphysics (contact mechanics + EQS finite volume)

| electrode | abs(Z) at 10 Hz | kappa (unloading) | kappa (lift-off) | mains diff / V_cm (Zin 10 MOhm) |
|---|---|---|---|---|
| wet | 21.5 kOhm | 0.001 | 0.596 | 1.26e-04 |
| dry | 442.9 kOhm | 0.398 | 0.624 | 1.47e-03 |

Verification: thin-strip analytic error 2.1e-04, reciprocity error 1.4e-10, grid-convergence spread 6.8e-05

## Simulator calibration (median-beat R²)

| bank | beats | median R² | IQR | R²>0.8 |
|---|---|---|---|---|
| full | 3907 | 0.951 | 0.91–0.97 | 0.94 |
| low10 | 390 | 0.952 | 0.91–0.97 | 0.94 |

## Denoising real ECG + real NSTDB noise (output SNR, dB)

| input SNR | R-peak F1 | band-pass | NN (no physics) | diff-sim | PINN λ=0.1 | PINN λ=1 |
|---|---|---|---|---|---|---|
| 0 dB | 0.98 | 6.269 ± 1.829 | 6.331 ± 1.805 | 7.700 ± 3.416 | 7.658 ± 2.863 | 6.928 ± 2.752 |
| 6 dB | 0.98 | 13.929 ± 5.357 | 13.600 ± 5.051 | 8.024 ± 3.471 | 12.150 ± 4.138 | 9.608 ± 3.317 |
| 12 dB | 0.99 | 20.954 ± 3.998 | 19.482 ± 2.779 | 8.445 ± 3.169 | 16.273 ± 3.209 | 11.730 ± 3.168 |

## Generative fidelity vs held-out real test set

| source | FED ↓ | MMD² ↓ | LSD dB ↓ | HR W1 ↓ | disc. score ↓ | TSTR AUROC ↑ |
|---|---|---|---|---|---|---|
| real_train_split | 4.37 | 0.0637 | 0.20 | 1.77 | 0.002 | 0.802 |
| diff_full | 72.88 | 0.1488 | 5.44 | 7.78 | 0.456 | 0.819 |
| diff_low10 | 107.25 | 0.2101 | 1.80 | 8.03 | 0.426 | 0.812 |
| diff_simpre_low10 | 43.23 | 0.0858 | 3.19 | 5.23 | 0.363 | 0.808 |
| sim_low10 | 6.11 | 0.0214 | 9.27 | 0.77 | 0.465 | 0.796 |
| sim_full | 6.62 | 0.0262 | 8.80 | 1.29 | 0.453 | 0.808 |

TRTR reference AUROC: 0.801
HR control MAE: 9.61 bpm (measured 57.1, 77.9, 95.4, 110.0 for requested 50/70/90/110)

## Downstream AUROC (NORM vs abnormal, mean ± std over seeds)

| regime | training data | clean | NSTDB 6 dB | NSTDB 0 dB |
|---|---|---|---|---|
| 10pct_real | real | 0.834 ± 0.002 | 0.794 ± 0.011 | 0.705 ± 0.030 |
| 10pct_real | real+sensor_model_aug | 0.809 ± 0.003 | 0.803 ± 0.006 | 0.790 ± 0.007 |
| 10pct_real | real+nstdb_noise_aug(reference) | 0.788 ± 0.004 | 0.785 ± 0.005 | 0.774 ± 0.006 |
| 10pct_real | real+simulator | 0.805 ± 0.010 | 0.774 ± 0.018 | 0.694 ± 0.028 |
| 10pct_real | real+diffusion | 0.813 ± 0.004 | 0.793 ± 0.012 | 0.754 ± 0.009 |
| 10pct_real | real+sim_pretrained_diffusion | 0.809 ± 0.007 | 0.790 ± 0.004 | 0.720 ± 0.008 |
| 10pct_real | real+sim_pretrained_diffusion+sensor_model_aug | 0.806 ± 0.008 | 0.804 ± 0.007 | 0.787 ± 0.005 |
| 100pct_real | real | 0.812 ± 0.009 | 0.762 ± 0.010 | 0.654 ± 0.009 |
| 100pct_real | real+sensor_model_aug | 0.833 ± 0.004 | 0.829 ± 0.004 | 0.818 ± 0.007 |
| 100pct_real | real+nstdb_noise_aug(reference) | 0.823 ± 0.009 | 0.820 ± 0.008 | 0.807 ± 0.010 |
| 100pct_real | real+simulator | 0.830 ± 0.008 | 0.808 ± 0.015 | 0.743 ± 0.021 |
| 100pct_real | real+diffusion | 0.835 ± 0.005 | 0.814 ± 0.014 | 0.756 ± 0.019 |
| 100pct_real | real+diffusion+sensor_model_aug | 0.836 ± 0.003 | 0.831 ± 0.002 | 0.827 ± 0.007 |

## Sensor-model priors: hand-set vs multiphysics-derived (AUROC)

| regime | augmentation | clean | NSTDB 6 dB | NSTDB 0 dB |
|---|---|---|---|---|
| 10pct_real | sensor_model_aug(hand-set priors) | 0.809 ± 0.003 | 0.803 ± 0.006 | 0.790 ± 0.007 |
| 10pct_real | sensor_model_aug(multiphysics priors) | 0.809 ± 0.004 | 0.803 ± 0.003 | 0.788 ± 0.000 |
| 100pct_real | sensor_model_aug(hand-set priors) | 0.833 ± 0.004 | 0.829 ± 0.004 | 0.818 ± 0.007 |
| 100pct_real | sensor_model_aug(multiphysics priors) | 0.819 ± 0.013 | 0.814 ± 0.014 | 0.801 ± 0.013 |

## Sampler / guidance sweep (selected on validation)

| model | sampler / w | saturated samples | FED (val) | LSD dB |
|---|---|---|---|---|
| diff_full | trailing/static/w=0 | 0.377 | 170.92 | 18.75 |
| diff_full | trailing/static/w=1 | 0.363 | 97.69 | 11.23 |
| diff_full | trailing/static/w=2 | 0.400 | 49.43 | 5.82 |
| diff_full | leading/static/w=0 | 0.000 | 1.61 | 0.38 |
| diff_full | leading/static/w=1 | 0.010 | 0.96 | 1.00 |
| diff_full | leading/static/w=2 | 0.023 | 1.52 | 1.78 |
| diff_full | leading/dynamic/w=0 | 0.000 | 1.74 | 0.39 |
| diff_full | leading/dynamic/w=1 | 0.003 | 0.84 | 0.59 |
| diff_full | leading/dynamic/w=2 | 0.020 | 1.30 | 1.26 |
| diff_simpre_low10 | trailing/static/w=0 | 0.200 | 135.78 | 18.29 |
| diff_simpre_low10 | trailing/static/w=1 | 0.223 | 66.23 | 9.26 |
| diff_simpre_low10 | trailing/static/w=2 | 0.247 | 29.12 | 3.33 |
| diff_simpre_low10 | leading/static/w=0 | 0.000 | 1.18 | 2.02 |
| diff_simpre_low10 | leading/static/w=1 | 0.003 | 2.74 | 0.36 |
| diff_simpre_low10 | leading/static/w=2 | 0.017 | 4.04 | 1.06 |
| diff_simpre_low10 | leading/dynamic/w=0 | 0.000 | 1.23 | 2.27 |
| diff_simpre_low10 | leading/dynamic/w=1 | 0.000 | 2.41 | 0.77 |
| diff_simpre_low10 | leading/dynamic/w=2 | 0.007 | 3.59 | 0.75 |

| model (selected w) | saturated | FED ↓ | MMD² ↓ | LSD dB ↓ | disc. score ↓ | TSTR AUROC ↑ |
|---|---|---|---|---|---|---|
| diff_full (leading/dynamic, w=1.0) | 0.005 | 1.28 | 0.0049 | 0.34 | 0.125 | 0.829 |
| diff_simpre_low10 (leading/static, w=0.0) | 0.002 | 2.01 | 0.0064 | 2.34 | 0.211 | 0.825 |

| regime | training data | clean | NSTDB 6 dB | NSTDB 0 dB |
|---|---|---|---|---|
| 10pct_real | real+sim_pretrained_diffusion(w_sel) | 0.824 ± 0.008 | 0.802 ± 0.002 | 0.750 ± 0.010 |
| 10pct_real | real+sim_pretrained_diffusion(w_sel)+sensor_model_aug | 0.807 ± 0.002 | 0.806 ± 0.004 | 0.784 ± 0.009 |
| 100pct_real | real+diffusion(w_sel) | 0.834 ± 0.007 | 0.811 ± 0.014 | 0.745 ± 0.028 |
| 100pct_real | real+diffusion(w_sel)+sensor_model_aug | 0.838 ± 0.001 | 0.834 ± 0.003 | 0.827 ± 0.003 |

## Independent finite-element cross-check of the electrode solver

P1 finite elements vs finite volume, wet electrode, 1 Hz–10 kHz: max |Z| difference 0.46 % (mean 0.10 %), phase within 0.14°; 20 % lift-off raises |Z| 11.92 % (FE) vs 11.92 % (FV).

## Real in-car ECG (PhysioNet drivedb): false cardiac alarms at 90 % validation specificity

18 drives, 7995 ten-second windows (1.00 with detectable beats).

| training | PTB-XL NORM false alarms | in-car, all | in-car, first 10 min | in-car, driving | PTB-XL AUROC |
|---|---|---|---|---|---|
| real | 0.124 ± 0.017 | 0.139 ± 0.013 | 0.122 ± 0.015 | 0.141 ± 0.013 | 0.812 ± 0.009 |
| real+sensor_model_aug | 0.110 ± 0.007 | 0.153 ± 0.009 | 0.124 ± 0.001 | 0.158 ± 0.010 | 0.833 ± 0.004 |
| real+nstdb_noise_aug(reference) | 0.094 ± 0.011 | 0.161 ± 0.013 | 0.137 ± 0.015 | 0.165 ± 0.013 | 0.823 ± 0.009 |
| real+diffusion(fixed)+sensor_model_aug | 0.103 ± 0.016 | 0.173 ± 0.002 | 0.146 ± 0.002 | 0.177 ± 0.002 | 0.838 ± 0.001 |

real+sensor_model_aug: change in in-car false alarms vs real-only +0.015 (95 % driver-bootstrap CI -0.006 to +0.041)

real+nstdb_noise_aug(reference): change in in-car false alarms vs real-only +0.022 (95 % driver-bootstrap CI -0.007 to +0.062)

real+diffusion(fixed)+sensor_model_aug: change in in-car false alarms vs real-only +0.035 (95 % driver-bootstrap CI -0.013 to +0.111)

## Physics-informed (limit-cycle) guidance for the diffusion sampler (no retraining)

Selected on validation: λ = 0.05

| sampler | HR-control MAE (bpm) ↓ | FED ↓ | MMD² ↓ | LSD dB ↓ | disc. ↓ | TSTR AUROC ↑ |
|---|---|---|---|---|---|---|
| unguided (λ=0) | 4.12 | 1.05 | 0.0030 | 0.37 | 0.142 | 0.827 |
| physics_guided (λ=0.05) | 3.98 | 2.64 | 0.0074 | 2.05 | 0.233 | 0.829 |

## Neural surrogate of the electrode simulator (held-out FE spectra)

| model | median abs(Z) error | p95 abs(Z) error | median phase error | 95 % coverage |
|---|---|---|---|---|
| quadratic_ridge_plain | 1.67 % | 7.09 % | 0.658° | – |
| quadratic_ridge_hybrid | 0.74 % | 3.50 % | 0.046° | – |
| analytic_contact_prior_only | 8.71 % | 47.24 % | 0.821° | – |
| ensemble_plain | 0.10 % | 0.39 % | 0.009° | 0.99 |
| ensemble_hybrid | 0.02 % | 0.24 % | 0.002° | 0.98 |

Speed: 37,714x faster than the FE solver per spectrum.

## Robust electrode design through the differentiable surrogate (verified by FE)

| design | w (mm) | s (mm) | FE objective J | SNR vs max-spacing | regret vs FE grid |
|---|---|---|---|---|---|
| surrogate_robust | 9.07 | 25.93 | -8.7017 | +1.21 dB | 0.0154 |
| surrogate_nominal_tissue | 9.08 | 25.92 | -8.7017 | +1.21 dB | 0.0153 |
| fe_grid_best | 11.00 | 24.00 | -8.6864 | +1.34 dB | 0.0000 |
| common_w10_s25 | 10.00 | 25.00 | -8.6914 | +1.30 dB | 0.0051 |
| max_spacing_w5_s30 | 5.00 | 30.00 | -8.8409 | +0.00 dB | 0.1545 |

Surrogate optimisation 12.8 s vs FE grid search 58 s.

## How much real data does synthetic data replace? (sensor-model augmentation in all runs)

| training data | clean AUROC | ≈ real data | NSTDB 0 dB AUROC | ≈ real data |
|---|---|---|---|---|
| 10 % real only | 0.809 ± 0.003 | – | 0.790 ± 0.007 | – |
| 25 % real only | 0.798 ± 0.002 | – | 0.781 ± 0.003 | – |
| 50 % real only | 0.808 ± 0.005 | – | 0.786 ± 0.005 | – |
| 100 % real only | 0.824 ± 0.011 | – | 0.802 ± 0.014 | – |
| 10 % real + 500 simulator | 0.791 ± 0.005 | 10 % | 0.780 ± 0.004 | 10 % |
| 10 % real + 1000 simulator | 0.784 ± 0.006 | 10 % | 0.773 ± 0.003 | 10 % |
| 10 % real + 2000 simulator | 0.785 ± 0.005 | 10 % | 0.771 ± 0.007 | 10 % |
| 10 % real + 4000 simulator | 0.799 ± 0.008 | 10 % | 0.783 ± 0.003 | 10 % |
| 10 % real + 8000 simulator | 0.792 ± 0.010 | 10 % | 0.779 ± 0.012 | 10 % |
| 10 % real + 500 sim_pretrained_diffusion | 0.807 ± 0.005 | 10 % | 0.781 ± 0.003 | 10 % |
| 10 % real + 1000 sim_pretrained_diffusion | 0.802 ± 0.008 | 10 % | 0.779 ± 0.011 | 10 % |
| 10 % real + 2000 sim_pretrained_diffusion | 0.793 ± 0.011 | 10 % | 0.771 ± 0.025 | 10 % |
| 10 % real + 4000 sim_pretrained_diffusion | 0.809 ± 0.010 | 50 % | 0.788 ± 0.008 | 10 % |
| 10 % real + 8000 sim_pretrained_diffusion | 0.822 ± 0.004 | 89 % | 0.803 ± 0.010 | 100 % |
