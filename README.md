<div align="center">

# Guardian Drive

**A research prototype for sensor-aware driving safety systems**

[![CI](https://github.com/AKilalours/guardian-drive/actions/workflows/ci.yml/badge.svg)](https://github.com/AKilalours/guardian-drive/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab)
![PyTorch](https://img.shields.io/badge/PyTorch-prototypes-ee4c2c)
![Status](https://img.shields.io/badge/status-research%20prototype-64748b)

[Watch the synthetic sensor replay](docs/media/guardian-synthetic-replay.mp4) ·
[Run the interactive dashboard locally](#run-the-interactive-dashboard) ·
[Explore the WESAD diffusion code](#conditional-diffusion-for-wesad)

<a href="docs/media/guardian-synthetic-replay.mp4"><img src="docs/media/guardian-synthetic-replay.gif" alt="Animated synthetic ECG, EDA, respiration, and acceleration traces from the deterministic Guardian Drive simulator" width="100%"></a>

<sub>The replay is generated from Guardian Drive's deterministic simulator. It contains synthetic signals only: no participant data and no model predictions.</sub>

</div>

---

## What this project is

Guardian Drive combines a deterministic multi-sensor simulator, a WebSocket dashboard, driving-perception experiments, and a conditional diffusion prototype for wearable time-series. It is a portfolio and research prototype, not a deployed vehicle safety system.

The most role-relevant work is the WESAD conditional 1D DDPM in [`generative/wesad_diffusion.py`](generative/wesad_diffusion.py). It prepares 10-second wrist-sensor windows, separates train and held-out participants, fits normalization on training participants, and supports classifier-free guidance and EMA sampling. The companion [`generative/evaluate_wesad_augmentation.py`](generative/evaluate_wesad_augmentation.py) defines a held-out evaluation comparing real-only training with real-plus-synthetic augmentation.

> **WESAD experiment status:** The code and CI smoke coverage are present, but full diffusion training and the held-out augmentation evaluation have not been run in this checkout. The raw WESAD dataset is separately licensed and was not found in the available project files. This repository reports no diffusion quality or downstream augmentation gain.

## Visual tour

<table>
  <tr>
    <td width="50%"><a href="docs/media/bev-slam-overview.png"><img src="docs/media/bev-slam-overview.png" alt="Static bird's-eye occupancy and SLAM overview from Guardian Drive project materials" width="100%"></a></td>
    <td width="50%"><a href="docs/media/safety-state-machine.png"><img src="docs/media/safety-state-machine.png" alt="Diagram of the nominal, advisory, caution, pullover, and escalation states" width="100%"></a></td>
  </tr>
  <tr>
    <td align="center"><sub>BEV and occupancy-map visual from the project materials. Static illustration; not a live camera feed.</sub></td>
    <td align="center"><sub>Safety-state design diagram; it does not represent a validated medical or vehicle-control policy.</sub></td>
  </tr>
</table>

## Run the interactive dashboard

The dashboard is interactive when run with the local WebSocket backend. The included scenarios use synthetic simulator inputs; they are useful for exercising the UI and pipeline, not for measuring real-world detection performance.

```bash
git clone https://github.com/AKilalours/guardian-drive.git
cd guardian-drive
python3 -m pip install -r requirements.txt
```

Start the backend in one terminal:

```bash
python3 backend_server.py
```

Start a scenario in a second terminal and leave it running. Open the dashboard from a third terminal:

```bash
python3 main.py --scenario drowsy --gps mock --no-webcam --duration 30
# in a third terminal, from the repository root
open guardian_dreamview_fresh.html
```

Try `normal`, `drowsy`, or `crash_severe` with `--scenario`. These are simulated scenarios, not real events or medical states.

## Conditional diffusion for WESAD

The denoiser is a class-conditional 1D U-Net for BVP, EDA, temperature, and three-axis acceleration. It includes participant-level splitting, train-only normalization, class balancing, classifier-free condition dropout/guidance, and exponential-moving-average weights.

Training and evaluation require a local WESAD copy obtained under its dataset terms:

```bash
python -m generative.wesad_diffusion \
  --wesad-root /path/to/WESAD \
  --epochs 50 \
  --samples-per-class 500

python -m generative.evaluate_wesad_augmentation \
  --wesad-root /path/to/WESAD \
  --checkpoint outputs/wesad_conditional_ddpm.pt \
  --synthetic outputs/wesad_conditional_ddpm_synthetic.npz \
  --synthetic-per-class 500
```

The evaluation script is designed to report per-held-out-subject balanced accuracy and macro-F1 with bootstrap intervals. Until it is run on the licensed data, there are no empirical claims here about fidelity, privacy, or downstream benefit. Generated signals must not be presented as real measurements.

## Evidence and limitations

- The CI workflow runs unit and smoke checks, validates tracked benchmark-result files, and checks the fusion weights.
- Individual experiment artifacts are under [`outputs/`](outputs/); read each artifact's protocol and source before quoting a metric.
- The earlier WESAD TCN window-level AUC of 0.9738 is flagged for subject leakage; the documented LOSO AUC is 0.6712 ([`policy/medical_guardrail.py`](policy/medical_guardrail.py)). These are baseline results, separate from the unrun diffusion experiment, and are not presented as deployment evidence.
- The deterministic generator in [`acquisition/simulator.py`](acquisition/simulator.py) is synthetic. It is not a physiological or multiphysics model.
- No clinical validation, prospective road testing, safety certification, or COMSOL multiphysics validation is claimed.

## Repository map

| Area | Path |
|---|---|
| Conditional wearable diffusion | [`generative/`](generative/) |
| Synthetic sensor simulator | [`acquisition/simulator.py`](acquisition/simulator.py) |
| Dashboard | [`guardian_dreamview_fresh.html`](guardian_dreamview_fresh.html), [`backend_server.py`](backend_server.py) |
| BEV perception and driving experiments | [`bev_perception/`](bev_perception/), [`evaluation/`](evaluation/) |
| CI smoke tests | [`tests/`](tests/), [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |
| Experiment artifacts | [`outputs/`](outputs/) |
| Safety notes | [`docs/SAFETY.md`](docs/SAFETY.md) |

## Rebuild the README replay

The embedded GIF and MP4 are rendered from `acquisition.simulator` with a fixed seed and are labeled synthetic. To regenerate them on macOS with Python dependencies installed, plus R base graphics and FFmpeg:

```bash
python3 scripts/make_readme_demo.py
```

## Safety note

This repository is for research and demonstration. It does not diagnose, predict, or treat medical conditions, and it must not be used to make clinical decisions or control a vehicle. Simulator labels and generated signals are illustrative only.

---

<div align="center">
<sub>Akila Lourdes Miriyala Francis · Akilan Manivannan</sub>
</div>
