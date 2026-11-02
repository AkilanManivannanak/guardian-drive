import torch

from generative.wesad_diffusion import (
    ConditionalDenoiser,
    DiffusionSchedule,
    UNCONDITIONAL_LABEL,
)


def test_conditional_and_unconditional_denoising_shapes():
    model = ConditionalDenoiser(width=16)
    samples = torch.randn(2, 6, 32)
    steps = torch.tensor([0, 1])
    labels = torch.tensor([0, UNCONDITIONAL_LABEL])

    prediction = model(samples, steps, labels)

    assert prediction.shape == samples.shape
    assert torch.isfinite(prediction).all()


def test_classifier_free_guided_sampling_returns_finite_windows():
    model = ConditionalDenoiser(width=16).eval()
    schedule = DiffusionSchedule(steps=2, device=torch.device("cpu"))
    labels = torch.tensor([0, 2])

    generated = schedule.sample(
        model, labels, shape=(2, 6, 32), guidance_scale=1.5
    )

    assert generated.shape == (2, 6, 32)
    assert torch.isfinite(generated).all()
