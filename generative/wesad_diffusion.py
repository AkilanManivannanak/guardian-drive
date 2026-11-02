"""Conditional DDPM for short, multimodal WESAD wrist-sensor windows.

This is a research prototype for synthetic-data experiments, not a medical
device or a claim that generated data improves downstream performance.
WESAD is distributed separately; see its license and provide a local path.
"""
from __future__ import annotations

import argparse
import copy
import math
import pickle
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


FS = 32
WINDOW_SECONDS = 10
WINDOW = FS * WINDOW_SECONDS
STRIDE = WINDOW // 2
CHANNELS = ("bvp", "eda", "temp", "acc_x", "acc_y", "acc_z")
CLASS_LABELS = (1, 2, 3)  # WESAD: baseline, stress, amusement
UNCONDITIONAL_LABEL = len(CLASS_LABELS)
DEFAULT_TRAIN = (2, 3, 4, 5, 6, 7, 8, 9, 14, 17)
DEFAULT_VALIDATION = (10, 11)
DEFAULT_TEST = (13, 15, 16)
# Retained for older evaluation commands and checkpoints.
DEFAULT_HELDOUT = DEFAULT_VALIDATION + DEFAULT_TEST


def _resample(x: np.ndarray, source_hz: int, target_hz: int = FS) -> np.ndarray:
    """Polyphase-resample the first axis; scipy is already used by this repo."""
    from scipy.signal import resample_poly

    divisor = math.gcd(source_hz, target_hz)
    return resample_poly(x, target_hz // divisor, source_hz // divisor, axis=0).astype(
        np.float32, copy=False
    )


def load_subject(root: Path, subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return wrist channels [time, 6] and aligned WESAD labels for one subject."""
    subject = f"S{subject_id}"
    candidates = (root / subject / f"{subject}.pkl", root / f"{subject}.pkl")
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing {subject}.pkl under {root}")

    # WESAD pickles must come from a trusted, locally obtained dataset archive.
    with path.open("rb") as stream:
        data = pickle.load(stream, encoding="latin1")
    wrist = data["signal"]["wrist"]
    bvp = _resample(np.asarray(wrist["BVP"]).reshape(-1, 1), 64)
    eda = _resample(np.asarray(wrist["EDA"]).reshape(-1, 1), 4)
    temp = _resample(np.asarray(wrist["TEMP"]).reshape(-1, 1), 4)
    acc = _resample(np.asarray(wrist["ACC"], dtype=np.float32).reshape(-1, 3), 32)
    channels = (bvp, eda, temp, acc)
    length = min(map(len, channels))
    signals = np.concatenate([item[:length] for item in channels], axis=1)

    source_labels = np.asarray(data["label"]).reshape(-1)
    # WESAD labels are sampled at 700 Hz. Pick the closest original label for
    # each target timestamp; this avoids interpolating categorical values.
    indices = np.minimum(
        np.rint(np.arange(length) * 700 / FS).astype(np.int64), len(source_labels) - 1
    )
    labels = source_labels[indices].astype(np.int64, copy=False)
    return signals, labels


def collect_windows(root: Path, subject_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    windows: List[np.ndarray] = []
    labels: List[int] = []
    for subject_id in subject_ids:
        signal, target = load_subject(root, subject_id)
        for start in range(0, len(signal) - WINDOW + 1, STRIDE):
            segment_labels = target[start : start + WINDOW]
            values, counts = np.unique(segment_labels, return_counts=True)
            dominant = int(values[np.argmax(counts)])
            # Keep only nearly pure affect windows; transitions and artifact
            # label 0 are excluded instead of being assigned a misleading class.
            if dominant not in CLASS_LABELS or counts.max() / WINDOW < 0.95:
                continue
            windows.append(signal[start : start + WINDOW])
            labels.append(dominant)
    if not windows:
        raise ValueError(f"No eligible windows found for subjects {list(subject_ids)}")
    return np.stack(windows).astype(np.float32), np.asarray(labels, dtype=np.int64)


def train_normalizer(train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True).clip(min=1e-5)
    return mean.astype(np.float32), std.astype(np.float32)


def sinusoidal_embedding(t: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    scale = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    phase = t.float().unsqueeze(1) * scale.unsqueeze(0)
    return torch.cat((phase.sin(), phase.cos()), dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embed_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.condition = nn.Linear(embed_dim, out_channels)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.condition(F.silu(embedding)).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return (h + self.skip(x)) / math.sqrt(2.0)


class ConditionalDenoiser(nn.Module):
    """Small 1D U-Net denoiser conditioned on diffusion time and WESAD class."""

    def __init__(self, channels: int = len(CHANNELS), width: int = 64, classes: int = 4):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)
        )
        self.class_embed = nn.Embedding(classes, width)
        self.input = nn.Conv1d(channels, width, 3, padding=1)
        self.down1 = ResidualBlock(width, width, width)
        self.downsample = nn.Conv1d(width, width * 2, 4, stride=2, padding=1)
        self.down2 = ResidualBlock(width * 2, width * 2, width)
        self.middle = ResidualBlock(width * 2, width * 2, width)
        self.upsample = nn.ConvTranspose1d(width * 2, width, 4, stride=2, padding=1)
        self.up = ResidualBlock(width * 2, width, width)
        self.output = nn.Sequential(
            nn.GroupNorm(8, width), nn.SiLU(), nn.Conv1d(width, channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, step: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embedding = self.time_mlp(sinusoidal_embedding(step, self.time_mlp[0].in_features))
        embedding = embedding + self.class_embed(labels)
        first = self.down1(self.input(x), embedding)
        second = self.down2(self.downsample(first), embedding)
        middle = self.middle(second, embedding)
        up = self.upsample(middle)
        return self.output(self.up(torch.cat((up, first), dim=1), embedding))


class DiffusionSchedule:
    def __init__(self, steps: int, device: torch.device):
        self.steps = steps
        beta = torch.linspace(1e-4, 0.02, steps, device=device)
        self.alpha_bar = torch.cumprod(1.0 - beta, dim=0)
        self.beta = beta
        self.alpha = 1.0 - beta

    def add_noise(self, clean: torch.Tensor, step: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bar[step].view(-1, 1, 1)
        return alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        labels: torch.Tensor,
        shape: Tuple[int, int, int],
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        x = torch.randn(shape, device=labels.device)
        for index in reversed(range(self.steps)):
            t = torch.full((len(labels),), index, device=labels.device, dtype=torch.long)
            beta, alpha = self.beta[index], self.alpha[index]
            conditional = model(x, t, labels)
            if guidance_scale == 1.0:
                prediction = conditional
            else:
                unconditional_labels = torch.full_like(labels, UNCONDITIONAL_LABEL)
                unconditional = model(x, t, unconditional_labels)
                prediction = unconditional + guidance_scale * (conditional - unconditional)
            x = (x - beta / (1 - self.alpha_bar[index]).sqrt() * prediction) / alpha.sqrt()
            if index > 0:
                x = x + beta.sqrt() * torch.randn_like(x)
        return x


def subject_ids(value: str) -> Tuple[int, ...]:
    return tuple(int(item.strip().lstrip("sS")) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wesad-root", type=Path, required=True, help="Local WESAD directory (not included)"
    )
    parser.add_argument("--train-subjects", default=",".join(map(str, DEFAULT_TRAIN)))
    parser.add_argument(
        "--validation-subjects", default=",".join(map(str, DEFAULT_VALIDATION)),
        help="Participants used only for denoising validation and model selection",
    )
    parser.add_argument(
        "--test-subjects", default=",".join(map(str, DEFAULT_TEST)),
        help="Participants reserved for final evaluation; never loaded during training",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--condition-dropout", type=float, default=0.1,
                        help="Probability of dropping class conditioning during training")
    parser.add_argument("--guidance-scale", type=float, default=1.5,
                        help="Classifier-free guidance scale for generated windows")
    parser.add_argument("--ema-decay", type=float, default=0.995,
                        help="Exponential moving average decay for denoiser weights")
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=0,
        help="Optionally generate this many normalized windows per WESAD class",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/wesad_conditional_ddpm.pt")
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_subjects = subject_ids(args.train_subjects)
    validation_subjects = subject_ids(args.validation_subjects)
    test_subjects = subject_ids(args.test_subjects)
    if any(len(items) != len(set(items)) for items in (train_subjects, validation_subjects, test_subjects)):
        parser.error("Subject lists must not contain duplicates")
    train_set, validation_set, test_set = map(
        set, (train_subjects, validation_subjects, test_subjects)
    )
    if train_set & validation_set or train_set & test_set or validation_set & test_set:
        parser.error("Training, validation, and test subject lists must be pairwise disjoint")
    if not 0.0 <= args.condition_dropout < 1.0:
        parser.error("--condition-dropout must be in [0, 1)")
    if args.guidance_scale < 0.0:
        parser.error("--guidance-scale must be non-negative")
    if not 0.0 <= args.ema_decay < 1.0:
        parser.error("--ema-decay must be in [0, 1)")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train, train_labels = collect_windows(args.wesad_root, train_subjects)
    validation, validation_labels = collect_windows(args.wesad_root, validation_subjects)
    mean, std = train_normalizer(train)
    train = np.clip((train - mean) / std, -5, 5)
    validation = np.clip((validation - mean) / std, -5, 5)
    # Labels 1..3 map to embedding indices 0..2.
    dataset = TensorDataset(
        torch.from_numpy(train).transpose(1, 2), torch.from_numpy(train_labels - 1)
    )
    class_counts = np.bincount(train_labels - 1, minlength=len(CLASS_LABELS)).clip(min=1)
    sampling_weights = 1.0 / class_counts[train_labels - 1]
    sampler = WeightedRandomSampler(
        torch.as_tensor(sampling_weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, drop_last=False)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device={device}")
    model = ConditionalDenoiser().to(device)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    schedule = DiffusionSchedule(args.diffusion_steps, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(args.epochs):
        model.train()
        losses: List[float] = []
        for clean, labels in loader:
            clean, labels = clean.to(device), labels.to(device)
            step = torch.randint(args.diffusion_steps, (len(clean),), device=device)
            noise = torch.randn_like(clean)
            noisy = schedule.add_noise(clean, step, noise)
            conditioned_labels = labels.clone()
            dropped = torch.rand(len(labels), device=device) < args.condition_dropout
            conditioned_labels[dropped] = UNCONDITIONAL_LABEL
            loss = F.mse_loss(model(noisy, step, conditioned_labels), noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
                    ema_parameter.mul_(args.ema_decay).add_(parameter, alpha=1.0 - args.ema_decay)
            losses.append(float(loss.detach().cpu()))
        print(f"epoch={epoch + 1:03d} train_noise_mse={np.mean(losses):.5f}")

    # A fixed, subject-disjoint validation loss checks denoising generalization.
    ema_model.eval()
    rng = np.random.default_rng(args.seed + 1)
    validation_tensor = torch.from_numpy(validation).transpose(1, 2).to(device)
    validation_target = torch.from_numpy(validation_labels - 1).to(device)
    count = min(2048, len(validation_tensor))
    indices = torch.as_tensor(rng.choice(len(validation_tensor), count, replace=False), device=device)
    validation_losses: List[float] = []
    with torch.no_grad():
        for batch_indices in indices.split(args.batch_size):
            clean, labels = validation_tensor[batch_indices], validation_target[batch_indices]
            step = torch.randint(args.diffusion_steps, (len(clean),), device=device)
            noise = torch.randn_like(clean)
            validation_losses.append(
                float(
                    F.mse_loss(
                        ema_model(schedule.add_noise(clean, step, noise), step, labels), noise
                    ).cpu()
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": ema_model.cpu().state_dict(),
            "mean": mean,
            "std": std,
            "channels": CHANNELS,
            "class_labels": CLASS_LABELS,
            "window_samples": WINDOW,
            "sample_rate_hz": FS,
            "diffusion_steps": args.diffusion_steps,
            "condition_dropout": args.condition_dropout,
            "guidance_scale": args.guidance_scale,
            "ema_decay": args.ema_decay,
            "train_subjects": train_subjects,
            "validation_subjects": validation_subjects,
            "test_subjects": test_subjects,
            "validation_noise_mse": float(np.mean(validation_losses)),
            "seed": args.seed,
        },
        args.output,
    )
    if args.samples_per_class > 0:
        generated, generated_labels = [], []
        with torch.no_grad():
            for class_index in range(len(CLASS_LABELS)):
                remaining = args.samples_per_class
                while remaining:
                    batch_count = min(args.batch_size, remaining)
                    labels = torch.full((batch_count,), class_index, device=device, dtype=torch.long)
                    batch = schedule.sample(
                        ema_model.to(device),
                        labels,
                        (batch_count, len(CHANNELS), WINDOW),
                        guidance_scale=args.guidance_scale,
                    )
                    generated.append(batch.cpu().numpy().transpose(0, 2, 1))
                    generated_labels.extend([CLASS_LABELS[class_index]] * batch_count)
                    remaining -= batch_count
        synthetic_path = args.output.with_name(args.output.stem + "_synthetic.npz")
        np.savez_compressed(
            synthetic_path,
            windows=np.concatenate(generated).astype(np.float32),
            labels=np.asarray(generated_labels, dtype=np.int64),
            channels=np.asarray(CHANNELS),
            normalized=True,
            normalization_mean=mean.reshape(-1),
            normalization_std=std.reshape(-1),
            sample_rate_hz=FS,
        )
        print(f"synthetic_windows={synthetic_path} (normalized; not clinical data)")
    print(f"windows_train={len(train)} windows_validation={len(validation)}")
    print(f"validation_subject_noise_mse={np.mean(validation_losses):.5f}")
    print(f"test_subjects_reserved={','.join(map(str, test_subjects))}")
    print(f"checkpoint={args.output}")


if __name__ == "__main__":
    main()
