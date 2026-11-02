# WESAD conditional diffusion: first held-out evaluation

**Run date:** 2026-11-02  
**Status:** exploratory result; not evidence of deployment readiness

## Protocol

The experiment trains the 1D class-conditional diffusion model on 10 WESAD participants (S2–S9, S14, S17), selects the EMA checkpoint using two separate validation participants (S10, S11), then evaluates once on three reserved participants (S13, S15, S16). Training uses 4,337 eligible 10-second windows; validation uses 889. The selected checkpoint was epoch 50 (seed 42; 1,000 cosine diffusion steps; 50-step deterministic DDIM sampling; 500 generated windows per class). Normalization statistics are fitted on training participants only.

The training script now evaluates EMA noise-prediction loss on the same fixed validation windows, timesteps, and noise at every epoch, then saves the checkpoint with the lowest validation loss. This makes the checkpoint-selection criterion reproducible; it does not guarantee good generated signals or downstream utility.

## Results

| Held-out diagnostic | Mean | Participant-bootstrap 95% interval |
|---|---:|---:|
| Absolute standardized channel-mean shift | 13.80 | 6.16–24.09 |
| Absolute channel standard-deviation ratio error | 42.54 | 11.18–85.23 |
| Mean channel Wasserstein distance | 2.14 | 1.99–2.25 |
| Log-spectral distance | 36.30 dB | 31.64–40.26 |
| Autocorrelation MAE, lags 1–32 | 0.283 | 0.258–0.309 |
| Cross-channel correlation RMSE | 0.269 | 0.239–0.294 |

In the fixed Logistic Regression utility probe, mean participant balanced accuracy was **0.462 real-only** and **0.456 with synthetic augmentation** (difference −0.005; 95% interval −0.099 to 0.056). Macro-F1 was **0.342 real-only** and **0.324 with augmentation** (difference −0.018; 95% interval −0.104 to 0.038). These results do not demonstrate a downstream benefit. The intervals use only three held-out participants and are consequently imprecise.

## Reproduce

Obtain WESAD directly under its [official terms](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html); do not put its raw files, generated windows, or model checkpoint in this repository. From the repository root, run:

```bash
python -m generative.wesad_diffusion \
  --wesad-root /path/to/WESAD \
  --epochs 50 --seed 42 \
  --diffusion-steps 1000 --sampling-steps 50 \
  --samples-per-class 500 \
  --output outputs/wesad_conditional_ddpm.pt

python -m generative.evaluate_wesad_fidelity \
  --wesad-root /path/to/WESAD \
  --checkpoint outputs/wesad_conditional_ddpm.pt \
  --synthetic outputs/wesad_conditional_ddpm_synthetic.npz \
  --synthetic-per-class 500

python -m generative.evaluate_wesad_augmentation \
  --wesad-root /path/to/WESAD \
  --checkpoint outputs/wesad_conditional_ddpm.pt \
  --synthetic outputs/wesad_conditional_ddpm_synthetic.npz \
  --synthetic-per-class 500
```

## Limits and next work

This is one seed, one small public dataset, and three held-out participants. The signals are wearable stress/affect data collected in a lab, not driving, fatigue, or human-behavior-at-scale data. The fidelity statistics are diagnostics without validated acceptance thresholds. The utility probe is a simple classifier experiment, not a drowsiness model. It does not establish privacy, memorization resistance, clinical validity, or safety. No claim of improved augmentation is made.

The large distributional errors point to further work before using generated windows as training data: inspect channel-specific normalization and conditioning, compare sampler and guidance settings on validation participants only, and evaluate with multiple seeds and a stronger task-specific protocol. Keep the reserved test participants untouched until those choices are fixed.

Dataset citation: P. Schmidt, A. Reiss, R. Duerichen, C. Marberger, and K. Van Laerhoven, “Introducing WESAD: A Multimodal Dataset for Wearable Stress and Affect Detection,” ICMI 2018, doi: [10.1145/3242969.3242985](https://doi.org/10.1145/3242969.3242985).
