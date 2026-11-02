"""Measure class-conditional signal fidelity on reserved WESAD participants.

Reports interpretable, univariate and temporal diagnostics for generated wrist
windows. These summaries are not a universal realism score and do not establish
clinical validity, privacy, or downstream usefulness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy.signal import welch
from scipy.stats import wasserstein_distance

from generative.wesad_diffusion import (
    CHANNELS,
    CLASS_LABELS,
    DEFAULT_HELDOUT,
    FS,
    WINDOW,
    collect_windows,
    subject_ids,
)


METRIC_NAMES = (
    "mean_abs_standardized_mean_shift",
    "mean_abs_std_ratio_error",
    "mean_channel_wasserstein",
    "mean_log_spectral_distance_db",
    "mean_autocorrelation_mae_lag_1_32",
    "cross_channel_correlation_rmse",
)


def normalize(windows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip((windows - mean) / np.maximum(std, 1e-5), -5, 5).astype(np.float32)


def _autocorrelation(windows: np.ndarray, max_lag: int = 32) -> np.ndarray:
    centered = windows - windows.mean(axis=1, keepdims=True)
    variance = np.mean(centered**2, axis=1).clip(min=1e-8)
    values = []
    for lag in range(1, max_lag + 1):
        covariance = np.mean(centered[:, :-lag] * centered[:, lag:], axis=1)
        values.append(np.mean(covariance / variance, axis=0))
    return np.stack(values, axis=0)


def compare_windows(real: np.ndarray, generated: np.ndarray) -> Dict[str, object]:
    """Compare normalized [window, time, channel] signals without pooling labels."""
    if (
        real.ndim != 3
        or generated.ndim != 3
        or real.shape[2] != len(CHANNELS)
        or real.shape[1:] != generated.shape[1:]
    ):
        raise ValueError("Expected real and generated arrays shaped [windows, time, 6]")
    if not len(real) or not len(generated):
        raise ValueError("Both real and generated sets must contain at least one window")
    real_mean = real.mean(axis=(0, 1))
    generated_mean = generated.mean(axis=(0, 1))
    real_std = real.std(axis=(0, 1)).clip(min=1e-6)
    generated_std = generated.std(axis=(0, 1))
    mean_shift = np.abs(generated_mean - real_mean) / real_std
    std_ratio_error = np.abs(generated_std / real_std - 1.0)
    channel_wasserstein = np.asarray(
        [
            wasserstein_distance(real[:, :, index].ravel(), generated[:, :, index].ravel())
            for index in range(len(CHANNELS))
        ]
    )

    segment_length = min(128, real.shape[1], generated.shape[1])
    real_freq, real_psd = welch(real, fs=FS, axis=1, nperseg=segment_length)
    generated_freq, generated_psd = welch(generated, fs=FS, axis=1, nperseg=segment_length)
    if not np.allclose(real_freq, generated_freq):
        raise ValueError("Real and generated windows must have matching time dimensions")
    mean_real_psd = real_psd.mean(axis=0)
    mean_generated_psd = generated_psd.mean(axis=0)
    spectral_distance = np.mean(
        np.abs(10.0 * np.log10(mean_generated_psd.clip(min=1e-12))
               - 10.0 * np.log10(mean_real_psd.clip(min=1e-12))),
        axis=0,
    )

    real_acf = _autocorrelation(real)
    generated_acf = _autocorrelation(generated)
    acf_error = np.mean(np.abs(generated_acf - real_acf), axis=0)
    real_corr = np.corrcoef(real.reshape(-1, len(CHANNELS)), rowvar=False)
    generated_corr = np.corrcoef(generated.reshape(-1, len(CHANNELS)), rowvar=False)
    real_corr = np.nan_to_num(real_corr, nan=0.0, posinf=0.0, neginf=0.0)
    generated_corr = np.nan_to_num(generated_corr, nan=0.0, posinf=0.0, neginf=0.0)
    cross_corr_rmse = float(np.sqrt(np.mean((generated_corr - real_corr) ** 2)))

    values = {
        "mean_abs_standardized_mean_shift": float(mean_shift.mean()),
        "mean_abs_std_ratio_error": float(std_ratio_error.mean()),
        "mean_channel_wasserstein": float(channel_wasserstein.mean()),
        "mean_log_spectral_distance_db": float(spectral_distance.mean()),
        "mean_autocorrelation_mae_lag_1_32": float(acf_error.mean()),
        "cross_channel_correlation_rmse": cross_corr_rmse,
    }
    values["per_channel"] = {
        channel: {
            "standardized_mean_shift": float(mean_shift[index]),
            "std_ratio_error": float(std_ratio_error[index]),
            "wasserstein": float(channel_wasserstein[index]),
            "log_spectral_distance_db": float(spectral_distance[index]),
            "autocorrelation_mae_lag_1_32": float(acf_error[index]),
        }
        for index, channel in enumerate(CHANNELS)
    }
    return values


def bootstrap_mean_interval(
    values: Sequence[float], seed: int, reps: int = 5000
) -> Optional[List[float]]:
    values_array = np.asarray(values, dtype=np.float64)
    if len(values_array) < 2:
        return None
    rng = np.random.default_rng(seed)
    samples = rng.choice(values_array, size=(reps, len(values_array)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def summarize_subjects(subject_values: Dict[str, List[Dict[str, object]]]) -> Dict[str, object]:
    """Average classes within person, then bootstrap participants as the unit."""
    output: Dict[str, object] = {}
    for metric_index, name in enumerate(METRIC_NAMES):
        subject_means = [
            float(np.mean([float(class_metrics[name]) for class_metrics in classes]))
            for classes in subject_values.values()
        ]
        output[name] = {
            "mean_across_subjects": float(np.mean(subject_means)),
            "participant_bootstrap_95pct_ci": bootstrap_mean_interval(subject_means, 100 + metric_index),
            "n_subjects": len(subject_means),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wesad-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--test-subjects", default=None)
    parser.add_argument("--synthetic-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/wesad_fidelity_eval.json"))
    args = parser.parse_args()
    if args.synthetic_per_class < 1:
        parser.error("--synthetic-per-class must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_subjects = set(checkpoint["train_subjects"])
    validation_subjects = set(checkpoint.get("validation_subjects", ()))
    configured_test_subjects = tuple(
        checkpoint.get("test_subjects", checkpoint.get("heldout_subjects", DEFAULT_HELDOUT))
    )
    evaluation_subjects = (
        subject_ids(args.test_subjects) if args.test_subjects else configured_test_subjects
    )
    if train_subjects.intersection(evaluation_subjects):
        parser.error("Evaluation participants overlap the generator training split")
    if validation_subjects.intersection(evaluation_subjects):
        parser.error("Evaluation participants overlap the model-selection validation split")
    if "test_subjects" in checkpoint and not set(evaluation_subjects).issubset(
        configured_test_subjects
    ):
        parser.error("Evaluation participants must come from the checkpoint's reserved test split")
    if len(evaluation_subjects) != len(set(evaluation_subjects)):
        parser.error("--test-subjects must not contain duplicates")

    with np.load(args.synthetic, allow_pickle=False) as archive:
        generated = archive["windows"].astype(np.float32)
        generated_labels = archive["labels"].astype(np.int64)
        channels = tuple(str(value) for value in archive["channels"].tolist())
        normalized = bool(archive["normalized"])
        stored_mean = archive["normalization_mean"].astype(np.float32).reshape(-1)
        stored_std = archive["normalization_std"].astype(np.float32).reshape(-1)
        sample_rate = int(archive["sample_rate_hz"])
    if channels != tuple(CHANNELS):
        parser.error(f"Expected channels {CHANNELS}; found {channels}")
    if not normalized:
        parser.error("Synthetic windows must declare normalized=true")
    if sample_rate != FS or generated.shape[1:] != (WINDOW, len(CHANNELS)):
        parser.error(f"Expected generated windows shaped [N, {WINDOW}, {len(CHANNELS)}] at {FS} Hz")
    checkpoint_mean = np.asarray(checkpoint["mean"]).reshape(-1)
    checkpoint_std = np.asarray(checkpoint["std"]).reshape(-1)
    if not np.allclose(stored_mean, checkpoint_mean) or not np.allclose(stored_std, checkpoint_std):
        parser.error("Synthetic normalization metadata does not match the checkpoint")

    rng = np.random.default_rng(args.seed)
    synthetic_by_class: Dict[int, np.ndarray] = {}
    for label in CLASS_LABELS:
        indices = np.flatnonzero(generated_labels == label)
        if len(indices) < args.synthetic_per_class:
            parser.error(
                f"Need {args.synthetic_per_class} generated windows for class {label}; found {len(indices)}"
            )
        selected = rng.choice(indices, args.synthetic_per_class, replace=False)
        synthetic_by_class[label] = generated[selected]

    per_subject: Dict[str, object] = {}
    class_metrics_by_subject: Dict[str, List[Dict[str, object]]] = {}
    for subject_id in evaluation_subjects:
        real_windows, real_labels = collect_windows(args.wesad_root, [subject_id])
        real_windows = normalize(real_windows, checkpoint["mean"], checkpoint["std"])
        subject_name = f"S{subject_id}"
        class_metrics_by_subject[subject_name] = []
        per_subject[subject_name] = {}
        for label in CLASS_LABELS:
            class_real = real_windows[real_labels == label]
            if not len(class_real):
                parser.error(f"No eligible windows for class {label} in {subject_name}")
            metrics = compare_windows(class_real, synthetic_by_class[label])
            per_subject[subject_name][str(label)] = {
                "real_windows": int(len(class_real)),
                "synthetic_windows": int(args.synthetic_per_class),
                **metrics,
            }
            class_metrics_by_subject[subject_name].append(metrics)

    by_class: Dict[str, object] = {}
    for label in CLASS_LABELS:
        metrics_by_subject = {
            subject: [per_subject[subject][str(label)]] for subject in per_subject
        }
        by_class[str(label)] = summarize_subjects(metrics_by_subject)

    report = {
        "protocol": "class-conditional signal diagnostics; participant-disjoint reserved test split",
        "train_subjects": sorted(train_subjects),
        "validation_subjects": sorted(validation_subjects),
        "evaluation_subjects": [f"S{item}" for item in evaluation_subjects],
        "channels": list(CHANNELS),
        "sample_rate_hz": FS,
        "window_samples": WINDOW,
        "synthetic_windows_per_class": args.synthetic_per_class,
        "metrics_are_diagnostics_not_acceptance_thresholds": True,
        "overall_macro_average_across_classes": summarize_subjects(class_metrics_by_subject),
        "by_class": by_class,
        "per_subject": per_subject,
        "limitations": [
            "Distributional similarity does not prove clinical validity or downstream utility.",
            "This report is not a privacy or memorization guarantee.",
            "Confidence intervals bootstrap participants; the small cohort limits precision.",
            "Metrics have no validated pass threshold and must be interpreted with plots and task-specific evaluation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
