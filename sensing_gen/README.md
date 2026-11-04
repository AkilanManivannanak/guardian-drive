# sensing_gen: simulation and generative models for wearable physiological signals

**[Interactive results explorer →](https://akilalours.github.io/guardian-drive/docs/explorer.html)** · [technical report (PDF)](../docs/sensing_gen_report.pdf)

The question this module answers: **can physics-based simulation and generative models give a
wearable-ECG model more, and harder, training data than real-world collection alone, and can we
measure whether they helped?**

Everything runs on public data. PTB-XL supplies real clinical ECG, used as a single lead (lead II
at 100 Hz) to mimic a wearable. MIT-BIH NSTDB supplies *real recorded* electrode-motion,
muscle-artifact and baseline-wander noise. Every number below comes from
`bash scripts/run_sensing_gen.sh` and is written to `outputs/sensing_gen/*.json`.

```
  [0] electrode-skin multiphysics (contact mechanics + EQS FV) ──► sensor-model priors

              real PTB-XL lead II (folds 1-8)               real NSTDB noise (held-out half)
                     │                                               │
        ┌────────────┼──────────────────────────────┐                │
        ▼            ▼                              ▼                ▼
  [1] differentiable  [3] conditional 1-D DDPM   [4] downstream classifier ──► AUROC on clean
      ECG ODE          (class + heart-rate FiLM,    trained on real ± synthetic      and noisy
      calibration      DDIM + CFG sampling)         ± sensor-model augmentation      test sets
        │                    ▲
        ▼                    │ pre-train on simulation, fine-tune on 10 % real
  physics prior ─────────────┘
  (ParamBank)  ──► [2] inverse PINN: denoise + identify ODE parameters
        │
        ▼
  electrode-skin sensor model (motion, EMG, mains, contact loss, ADC) ──► augmentation
```

## Components

| file | what it is |
|---|---|
| `physics/ecg_ode.py` | McSharry et al. (2003) ECG dynamical model in PyTorch. The RR process has Mayer-wave and respiratory-sinus-arrhythmia components, plus respiratory baseline and amplitude coupling and an AF-like irregular mode. The linear z-equation is solved **exactly** with an exponential integrator implemented as an FFT convolution. That makes it batched and differentiable w.r.t. all 15 wave parameters, and it is checked against an RK4 reference in the tests. |
| `multiphysics/electrode_skin_fv.py` | **Electro-mechanical multiphysics model** of a biopotential electrode on layered tissue (epidermis/dermis, fat, muscle). Contact mechanics (asperity contact area for dry electrodes, elastic gel and stratum-corneum compression, partial lift-off) is coupled to an **electro-quasi-static finite-volume solver** for div(σ* grad φ) = 0 with complex conductivity. The solver is verified against an analytic thin-strip solution, reciprocity and grid convergence. It predicts wet vs dry impedance spectra, motion sensitivity and mains conversion, which set the sensor model's priors. |
| `physics/calibrate.py` | Fits the ODE to thousands of real median beats in parallel **by back-propagating through the simulator**. The fitted parameter vectors form a class-conditional empirical prior that the simulator resamples from. |
| `physics/sensor.py` | Lumped-element model of the electrode–skin interface in a moving vehicle. Road and body events drive a damped seat/torso response. That motion produces a skin-stretch potential (AC-coupled), modulates electrode impedance (which changes ECG gain through the voltage divider), and causes impedance-mismatch mains pickup (60 Hz aliased to 40 Hz). The model also covers respiration-locked baseline wander, burst EMG, electrode lift-off to the rail, and ADC quantisation and clipping. It is fully batched with FFT filters, so it runs inside the training loop. |
| `pinn/ecg_pinn.py` | **Inverse PINN.** A Fourier-feature MLP for the clean ECG, a low-order baseline model, and learnable ODE parameters, trained with data fit + autograd ODE residual at stochastic collocation points. Many segments are solved in parallel with batched (einsum) MLP weights. It includes a hard-constraint differentiable-simulation ablation. |
| `diffusion/` | 1-D U-Net (FiLM conditioning on diffusion time, diagnostic class and heart rate, self-attention at the bottleneck), cosine-schedule DDPM, DDIM sampler, classifier-free guidance, EMA. |
| `eval/` | Fréchet ECG distance and MMD² on classifier embeddings, Welch log-spectral distance, heart-rate Wasserstein distance, discriminative score, train-on-synthetic-test-on-real (TSTR), and the downstream train/eval loop. |
| `multiphysics/electrode_skin_fem.py` | Independent P1 finite-element solver (contact-impedance Robin condition) used to cross-check the finite-volume solver; `fem_spectrum` assembles once and re-solves per frequency for dataset generation. |
| `surrogate/` | **Neural surrogate** of the FE electrode simulator: parameter space and Latin-hypercube sampling, parallel FE dataset generation, a physics-informed (analytic contact prior + learned correction) deep ensemble with β-NLL variance heads, and robust electrode design by projected gradient ascent through the surrogate. |
| `behavior/` | Second modality: UCI HAR loader (raw 6-axis IMU), a physics IMU sensor model (rigid re-orientation of accelerometer and gyroscope, calibration errors, loose-mount wobble, clock mismatch), a multi-channel activity classifier and a class-conditional IMU diffusion model. |
| `api.py`, `cli.py` | Public API and `sensing-gen` command line for the sensor model, generator and surrogate; see `docs/MODEL_CARD.md`. |
| `diffusion/physics_guidance.py` | Limit-cycle guidance energy from the ECG ODE, applied to the sampler's x̂₀ estimate (an ablation; see results §8). |
| `experiments/` | `run_multiphysics`, `run_calibration`, `run_pinn`, `run_diffusion`, `run_downstream`, `run_sensor_ablation`, `run_guidance`, `run_driving`, `run_fem_crosscheck`, `run_physics_guidance`, `run_surrogate`, `run_design`, `run_scaling`, orchestrated by `run_all` (with `smoke` and `full` profiles). |

## Protocol (no leakage)

* PTB-XL recommended split: folds 1–8 train, 9 validation (model selection), 10 test.
* Low-data regime: a stratified 10 % of the training records. In this regime the simulator bank, the
  diffusion models and the classifier only ever see those records.
* NSTDB noise: every recording is split in time. The **first half** is the only noise any training
  configuration may use (the "reference" augmentation). The **second half** is used only to corrupt
  the test set. The physics sensor model never sees NSTDB at all.
* Downstream numbers are mean ± std over 3 seeds. Model selection uses the clean validation set only.

## Results

All numbers come from `outputs/sensing_gen/*.json`, regenerated by the commands below. The full tables are in
[`outputs/sensing_gen/RESULTS.md`](../outputs/sensing_gen/RESULTS.md), and the write-up is in
[`docs/sensing_gen_report.pdf`](../docs/sensing_gen_report.pdf).

### 1. Physics-based sensor simulation makes the classifier robust to real sensor noise

AUROC, NORM vs abnormal, mean ± std over 3 seeds. The test set is corrupted with real NSTDB noise that no training run ever sees.

| training data (100 % real) | clean | NSTDB 6 dB | NSTDB 0 dB |
|---|---|---|---|
| real only | 0.812 ± 0.009 | 0.762 ± 0.010 | 0.654 ± 0.009 |
| + real NSTDB noise (train half; reference) | 0.823 ± 0.009 | 0.820 ± 0.008 | 0.807 ± 0.010 |
| **+ physics sensor model** | 0.833 ± 0.004 | 0.829 ± 0.004 | **0.818 ± 0.007** |
| + calibrated-simulator records | 0.830 ± 0.008 | 0.808 ± 0.015 | 0.743 ± 0.021 |
| + diffusion samples (original sampler) | 0.835 ± 0.005 | 0.814 ± 0.014 | 0.756 ± 0.019 |
| + diffusion samples + sensor model (original sampler) | 0.836 ± 0.003 | 0.831 ± 0.002 | 0.827 ± 0.007 |
| **+ diffusion samples + sensor model (fixed sampler)** | **0.838 ± 0.001** | **0.834 ± 0.003** | **0.827 ± 0.003** |

The sensor model **never sees NSTDB**, yet it beats augmenting with the real recorded noise itself.
With 10 % of the real data it lifts 0 dB AUROC from 0.705 to 0.790. It costs about 0.025 clean AUROC in that regime, which is a real trade-off and is reported as one.

### 2. Conditional diffusion: a sampler bug found, diagnosed and fixed

**Diagnosis.** The first full run produced realistic-looking but unstable samples: about 40 % of them
were pinned to the ±12 clip rail. The obvious suspect was classifier-free guidance, but a sweep showed the
failure persists at w = 0 (39 % saturated). The real cause was the sampler. DDIM started at t = T−1,
where the cosine schedule has ᾱ ≈ 2.4×10⁻⁹, so the x̂₀ estimate amplifies ε-errors by about 2×10⁴. Starting
from the standard "leading" timestep (ᾱ ≈ 9×10⁻⁴, ×34) and using per-sample dynamic thresholding fixes it
**without retraining**. The sampler and guidance were selected on the validation fold; the numbers below
are on the held-out test fold (`guidance_results.json`, `fig_samples_wsel.png`).

| validation sweep (full-data model) | saturated samples | Fréchet ECG dist. |
|---|---|---|
| original sampler (trailing / static clip), best w | 40.0 % | 49.4 |
| leading timesteps, static clip, w = 1 | 1.0 % | 0.96 |
| **leading timesteps + dynamic threshold, w = 1 (selected)** | **0.3 %** | **0.84** |

| test fold, fixed sampler | saturated | Fréchet ECG dist. ↓ | MMD² ↓ | log-spectral dist. dB ↓ | real-vs-synthetic discriminator acc. (0.5 = indistinguishable) | TSTR AUROC ↑ |
|---|---|---|---|---|---|---|
| diffusion, 100 % real | 0.5 % | 1.28 | 0.0049 | 0.34 | **0.625** (was 0.956) | **0.829** |
| diffusion, simulation-pretrained + 10 % real | 0.2 % | 2.01 | 0.0064 | 2.34 | 0.711 | 0.825 |

* A classifier trained **only on generated ECG** reaches **0.829 AUROC**; one trained on the real training data gets 0.801.
* Heart-rate conditioning: MAE **4.1 bpm** with the fixed sampler (320 test samples requested at 50/70/90/110 bpm; `physics_guidance_results.json`). The original sampler gave 9.6 bpm (57/78/95/110 measured).
* With the fixed sampler, diffusion samples + sensor-model augmentation give **0.838 ± 0.001 clean / 0.834 ± 0.003 at 6 dB / 0.827 ± 0.003 at 0 dB**, the best result in the study.
* With the original (broken) sampler, simulation pre-training reduced the 10 %-data model's Fréchet distance 2.5× (107 → 43). That comparison was not repeated with the fixed sampler, because the 10 %-real-only model was not re-scored. Treat it as indicative only.

Original-sampler fidelity table, for the record:

| source (vs held-out real test) | Fréchet ECG dist. ↓ | log-spectral dist. dB ↓ | discriminator acc. | TSTR AUROC ↑ |
|---|---|---|---|---|
| real (train split) | 4.4 | 0.20 | 0.50 | 0.802 |
| calibrated physics simulator | 6.6 | 8.80 | 0.95 | 0.808 |
| diffusion, 100 % real (original sampler) | 72.9 | 5.44 | 0.96 | 0.819 |
| diffusion, 10 % real (original sampler) | 107.3 | 1.80 | 0.93 | 0.812 |
| diffusion, sim-pretrained + 10 % (original sampler) | 43.2 | 3.19 | 0.86 | 0.808 |

The embedding network behind the Fréchet / MMD scores is retrained per experiment with identical settings,
so compare distances within a table, not across tables.

### 3. Calibrated physics and multiphysics

* The ECG ODE, calibrated by gradient descent through the simulator, explains **3,907 real median beats** with median **R² = 0.951** (IQR 0.91–0.97); see `fig_calibration.png`.
* The electrode–skin multiphysics solver is verified to **2×10⁻⁴** (analytic thin strip), **10⁻¹⁰** (reciprocity) and **7×10⁻⁵** (grid convergence).
  It predicts 21.5 kΩ (gel) vs 443 kΩ (dry) at 10 Hz, near-zero pressure sensitivity for gel but κ ≈ 0.4 for dry electrodes, and lift-off sensitivity κ ≈ 0.6.
* **Independent finite-element check:** `multiphysics/electrode_skin_fem.py` (P1 triangles, contact-impedance Robin condition) agrees with the finite-volume solver to 0.46 % max / 0.10 % mean in |Z| over 1 Hz–10 kHz, with phase within 0.14° and an identical 11.9 % rise at 20 % lift-off (`fem_crosscheck.json`, `fig_fem_crosscheck.png`).
<!-- COMSOL-CHECK:BEGIN -->
* **COMSOL:** the configuration can be rebuilt in COMSOL with the [step-by-step guide](multiphysics/comsol/README.md) and compared with `python -m sensing_gen.multiphysics.compare_comsol`. Results obtained under COMSOL evaluation licences are not published here, per their terms.
<!-- COMSOL-CHECK:END -->
* **Negative result.** Widening the sensor model's priors to the multiphysics range did *not* help (0 dB AUROC 0.801 ± 0.013 vs 0.818 ± 0.007). The augmentation benefit is robust to the exact priors.

### 4. Real in-car ECG: a deployment check that came back null

PhysioNet *drivedb* (Healey & Picard, 2005) contains ECG from 18 drives on real Boston roads: 7,995 ten-second windows. No diagnoses exist, so every abnormal output is counted as a **false cardiac alarm**. Thresholds are set to 90 % specificity on PTB-XL validation; nothing is tuned on driving data (`driving_results.json`, `fig_driving.png`).

| training (3 seeds) | false alarms, PTB-XL NORM test | false alarms, in car (all) | in car, while driving |
|---|---|---|---|
| real only | 12.4 % | **13.9 %** | 14.1 % |
| + physics sensor model | 11.0 % | 15.3 % | 15.8 % |
| + real NSTDB noise (reference) | 9.4 % | 16.1 % | 16.5 % |
| + diffusion (fixed) + sensor model | 10.3 % | 17.3 % | 17.7 % |

* **The models transfer.** In-car false alarms (13.9 %) sit close to the clinical design rate (12.4 % on PTB-XL), even though drivedb uses a different chest-lead placement and an older recorder.
* **The robustness gains do not show up here.** Every augmentation leaves in-car false alarms unchanged within noise; the driver-bootstrap 95 % CIs all include zero (sensor model +1.5 pp, CI −0.6 to +4.1). The drivedb ECG is clean (beats detectable in 99.9 % of windows), so the motion-artifact robustness measured on NSTDB has nothing to act on.
* **False alarms come from a few drivers.** The median driver has a 0.5–1.6 % false-alarm rate against 14–17 % overall. That points to electrode placement and morphology shift, not noise, as the remaining sim-to-real gap. Simulating lead placement, or calibrating per user, is the next step.

### 5. Inverse PINN: where physics helps and where it does not

| input SNR (real ECG + real NSTDB noise) | band-pass 0.5–40 Hz | NN, no physics | diff-sim (hard) | PINN λ=0.1 |
|---|---|---|---|---|
| 0 dB | 6.27 | 6.33 | **7.70** | 7.66 |
| 6 dB | **13.93** | 13.60 | 8.02 | 12.15 |
| 12 dB | **20.95** | 19.48 | 8.44 | 16.27 |

The values are output SNR in dB on 16 test records.
On simulated data with known parameters, the PINN recovers P/T-wave strength within 4–9 % and timing within 0.03 rad, and raises SNR from 9.0 to 11.7 dB. QRS strength is poorly identifiable at 100 Hz (35 % error).
On real data the physics prior pays off only when noise dominates. At higher SNR, a five-event single-template ODE cannot represent all real morphology, so plain filtering wins.

### 6. Neural surrogate of the electrode simulator

The FE solver (`fem_spectrum`) generates 4,000 training, 1,000 test and 300 out-of-range impedance spectra (21 frequencies, 1 Hz–10 kHz) over 10 parameters sampled by Latin hypercube: dermis and fat thickness; dermis, fat and muscle conductivity; stratum-corneum conductivity and permittivity (log-uniform); electrode width and spacing; lift-off. Targets are log10|Z| and phase. Models are 5-member MLP deep ensembles with mean and log-variance heads (MSE warm-up, then β-NLL, β = 0.5). The **hybrid** model predicts only the correction to the analytic contact impedance Z_c = (1/w_A + 1/w_B)/(L·Y_sc).

| model (1,000 held-out spectra) | median \|Z\| error | p95 \|Z\| error | median phase error | 95 % interval coverage |
|---|---|---|---|---|
| analytic contact prior only | 8.71 % | 47.24 % | 0.821° | – |
| quadratic ridge, plain / hybrid | 1.67 % / 0.74 % | 7.09 % / 3.50 % | 0.658° / 0.046° | – |
| deep ensemble, plain | 0.10 % | 0.39 % | 0.009° | 0.99 |
| **deep ensemble, hybrid** | **0.02 %** | **0.24 %** | **0.002°** | 0.98 |

| FE training simulations | 250 | 500 | 1,000 | 2,000 | 4,000 |
|---|---|---|---|---|---|
| plain ensemble, median \|Z\| error | 0.46 % | 0.20 % | 0.13 % | 0.11 % | 0.10 % |
| hybrid ensemble | **0.11 %** | **0.05 %** | **0.03 %** | **0.02 %** | **0.02 %** |

* The hybrid error matches the FE discretisation floor (coarse vs fine mesh, 0.015 % median), so more data cannot help further without a finer solver.
* Out of range (each range widened 25 %): error 0.16 % with ensemble σ 4.1× larger; σ and actual error are rank-correlated (`fig_surrogate.png`).
* 325 ms per FE spectrum vs 8.6 µs per surrogate spectrum (batched, CPU): 37,714× faster.

### 7. Robust electrode design through the surrogate

Objective: J(w, s) = E[ln s − ½ ln Re Z(10 Hz)] over 512 tissue/lift-off states (signal grows with spacing, thermal noise with √Re Z; +1 in J is +8.7 dB), with s + w ≤ 35 mm and s − w ≥ 5 mm. Projected Adam on (w, s) from three starts, back-propagating through the surrogate; every design is then re-scored with the FE solver on 48 *independent* tissues.

| design | w, s (mm) | FE objective J | vs FE grid optimum |
|---|---|---|---|
| FE brute-force grid (66 designs) | 11.0, 24.0 | −8.686 | – |
| surrogate gradient optimum | 9.1, 25.9 | −8.702 | −0.13 dB |
| common layout | 10.0, 25.0 | −8.691 | −0.04 dB |
| "maximise spacing" heuristic | 5.0, 30.0 | −8.841 | −1.34 dB |

The surrogate finds the right region in 13 s; the optimum is flat, so its 0.13 dB regret reflects surrogate error at the 10⁻⁴ level of J. The robust and nominal-tissue optima coincide. The design lesson is the heuristic's 1.3 dB loss.

### 8. Physics-informed limit-cycle guidance (negative result)

Guidance energy E = ¼ Σ (x(t) − x(t+τ))², τ = 60 fs/HR, applied as x̂₀ ← x̂₀ − λ(x̂₀ − P_τ x̂₀) at every DDIM step. λ ∈ {0, 0.05, 0.1, 0.2, 0.4} swept on validation; λ = 0.05 selected (Fréchet 0.84 → 0.50 on 300 validation samples).
On the test fold it did **not** hold: Fréchet 1.05 → 2.64, log-spectral distance 0.37 → 2.05 dB, HR-control MAE 4.12 → 3.98 bpm, TSTR 0.827 → 0.829. λ ≥ 0.2 collapses fidelity (Fréchet 12–32). Averaging beats one nominal period apart blurs the QRS because real RR intervals vary (RR std ≈ 120 ms). A phase-aware projection (aligning to detected beats) would be the next attempt.

### 9. How much real data does synthetic data replace?

10 % real training data plus N synthetic records, against real-only training at 10/25/50/100 %; sensor-model augmentation in every run; 3 seeds; generators trained on the 10 % subset only (`scaling_results.json`, `fig_scaling.png`).

| training data | clean AUROC | 0 dB real-noise AUROC |
|---|---|---|
| 10 % / 25 % / 50 % / 100 % real only | 0.809 / 0.798 / 0.808 / 0.824 | 0.790 / 0.781 / 0.786 / 0.802 |
| 10 % real + 500 / 1k / 2k / 4k / 8k physics-simulator ECG | 0.791 / 0.784 / 0.785 / 0.799 / 0.792 | 0.780 / 0.773 / 0.771 / 0.783 / 0.779 |
| 10 % real + 500 / 1k / 2k / 4k / **8k** diffusion ECG | 0.807 / 0.802 / 0.793 / 0.809 / **0.822** | 0.781 / 0.779 / 0.771 / 0.788 / **0.803** |

* With 8,000 generated records, 10 % real data matches 100 % real data (0.822 vs 0.824 clean; 0.803 vs 0.802 at 0 dB). Smaller generated sets do not, and simulator samples never help here.
* The real-only curve is flat from 10 % to 50 % once sensor augmentation is on, and the seed spread is about ±0.01. Read this as "generation at scale recovers the last real-data gain", not as an exact exchange rate.

### 10. Second modality: phone IMU activity recognition (human behaviour)

UCI HAR raw inertial signals (total acceleration incl. gravity, angular rate; 128 samples at 50 Hz; 6 activities). Train on 17 subjects, select on 4, test on the 9 official test subjects. Test sets: as recorded, and every window rotated by a uniform random 3-D rotation applied jointly to accelerometer and gyroscope (what a re-oriented rigid phone measures). 3 seeds; macro-F1.

| regime | training data | as recorded | re-oriented phone |
|---|---|---|---|
| 100 % | real | 0.919 ± 0.006 | 0.163 ± 0.008 |
| 100 % | + orientation-only augmentation | 0.962 ± 0.003 | 0.795 ± 0.014 |
| 100 % | + physics IMU sensor model | **0.966 ± 0.002** | **0.795 ± 0.011** |
| 100 % | + diffusion samples | 0.957 ± 0.002 | 0.288 ± 0.009 |
| 100 % | + diffusion + sensor model | 0.958 ± 0.013 | 0.681 ± 0.032 |
| 10 % | real | 0.892 ± 0.009 | 0.192 ± 0.005 |
| 10 % | + orientation-only augmentation | 0.893 ± 0.003 | 0.608 ± 0.062 |
| 10 % | + physics IMU sensor model | 0.910 ± 0.008 | 0.623 ± 0.061 |
| 10 % | + diffusion samples | 0.890 ± 0.010 | 0.279 ± 0.015 |
| 10 % | + diffusion + sensor model | 0.928 ± 0.020 | 0.644 ± 0.066 |

Generator (trained on its regime only): Fréchet distance in a real-trained classifier's embedding 114.8 (100 %) and 333.6 (10 %), against 2.7 / 3.6 for held-out real windows; label consistency 0.55 / 0.46; train-on-synthetic macro-F1 0.945 / 0.777. The samples carry class information but have too much high-frequency content (visible in `fig_behavior.png`); a fidelity fix (as for the ECG sampler) is the next step.

## Reproduce

```bash
bash scripts/run_sensing_gen.sh smoke   # download + tests + 3-minute smoke pass
bash scripts/run_sensing_gen.sh full    # full run (~4 h on an Apple M-series laptop, MPS)
python -m sensing_gen.data.download --drivedb-only
python -m sensing_gen.experiments.run_all multiphysics sensor_ablation guidance driving   # follow-up steps
python -m sensing_gen.experiments.run_all fem_crosscheck physics_guidance surrogate design scaling   # surrogate, design, scaling
python -m sensing_gen.experiments.run_all behavior   # phone IMU (downloads UCI HAR, ~60 MB)
python -m pytest -q tests/test_sensing_gen.py
```

## Limitations

* PTB-XL is resting, clinical 12-lead ECG. A single lead from it approximates, but is not, a wrist or
  chest-strap wearable. NSTDB noise is real, but it was recorded on stationary subjects.
* The multiphysics model is 2-D (per unit electrode length) and uses nominal literature tissue
  properties. Contact mechanics is treated quasi-statically with closed-form layer compression, not
  a full 3-D elasticity solve. It is written from scratch in Python/SciPy (finite volume, with an
  independent finite-element check), not COMSOL. The workflow (geometry → coupled physics → verified solve → parameter sweep → export to
  the ML pipeline) is the same one a COMSOL model would sit in.
* The ODE describes one lead with five Gaussian events. Beat-to-beat morphology changes and
  multi-lead geometry are outside the model, which is exactly where the diffusion model and the
  PINN's soft constraint take over.
