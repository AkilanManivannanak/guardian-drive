"""Unit tests for sensing_gen (numpy-only tests always run; torch tests skip without torch)."""
import math

import numpy as np
import pytest

from sensing_gen.data.datasets import corrupt_at_snr
from sensing_gen.data.signal_utils import detect_rpeaks, heart_rate, snr_db
from sensing_gen.data.wfdb_io import decode_212, encode_212, read_record
from sensing_gen.eval.fidelity import frechet_distance, mmd_rbf
from sensing_gen.eval.metrics import auroc

try:
    import torch
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# ---------------------------------------------------------------- data / signal processing
def test_format212_roundtrip():
    rng = np.random.default_rng(0)
    v = rng.integers(-2048, 2048, 1001)
    out = decode_212(np.frombuffer(encode_212(v), dtype=np.uint8), len(v))
    assert np.array_equal(out, v)


def test_read_record_format16(tmp_path):
    sig = (np.arange(200).reshape(100, 2) - 50).astype("<i2")
    (tmp_path / "r.dat").write_bytes(sig.tobytes())
    (tmp_path / "r.hea").write_text("r 2 100 100\nr.dat 16 1000.0(0)/mV 16 0 0 0 0 I\n"
                                    "r.dat 16 1000.0(0)/mV 16 0 0 0 0 II\n")
    x, hdr = read_record(tmp_path / "r")
    assert x.shape == (100, 2) and hdr.fs == 100
    assert np.allclose(x, sig / 1000.0)
    assert hdr.signals[1].description == "II"


def spike_train(hr=75, fs=100, n=1000, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    period = 60 / hr * fs
    peaks = np.arange(period / 2, n - 3, period).astype(int)
    for p in peaks:
        x[p - 2 : p + 3] += np.array([0.2, 0.6, 1.0, 0.6, 0.2])
    return x + noise * rng.standard_normal(n), peaks


def test_rpeak_detector_and_hr():
    x, peaks = spike_train(hr=72, noise=0.05)
    rp = detect_rpeaks(x)
    assert abs(len(rp) - len(peaks)) <= 1
    hr, _ = heart_rate(x)
    assert abs(hr - 72) < 2


def test_corrupt_at_snr_exact():
    rng = np.random.default_rng(1)
    clean, noise = rng.standard_normal((4, 1000)), rng.standard_normal((4, 1000)) * 7
    for target in (0, 6, 12):
        y = corrupt_at_snr(clean, noise, target)
        for c, yy in zip(clean, y):
            assert abs(snr_db(c, yy) - target) < 1e-3


def test_auroc_matches_definition():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    s = rng.standard_normal(300) + y
    brute = np.mean([[float(a > b) + 0.5 * float(a == b) for b in s[y == 0]] for a in s[y == 1]])
    assert abs(auroc(y, s) - brute) < 1e-9
    assert auroc(y, y.astype(float)) == 1.0


def test_fidelity_metrics_sanity():
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal((500, 8)), rng.standard_normal((500, 8))
    c = rng.standard_normal((500, 8)) + 1.5
    assert frechet_distance(a, b) < frechet_distance(a, c)
    assert abs(mmd_rbf(a, b)) < 0.01 < mmd_rbf(a, c)


# ---------------------------------------------------------------- physics
@needs_torch
def test_exponential_integrator_matches_rk4():
    from sensing_gen.physics.ecg_ode import default_params, forcing, integrate_exp, integrate_rk4
    p = default_params(2)
    dt = 1 / 2000
    t = torch.arange(0, 3.0, dt)
    phase = 2 * math.pi * t[None].repeat(2, 1) / torch.tensor([[0.8], [1.0]])
    z0 = torch.zeros_like(phase)
    z_exp = integrate_exp(forcing(phase, p) + z0, dt)
    z_rk = integrate_rk4(phase, p, z0, dt)
    err = (z_exp - z_rk).abs().max() / z_rk.abs().max()
    assert err < 0.02


@needs_torch
def test_simulator_heart_rate_and_shape():
    from sensing_gen.physics.ecg_ode import ECGSimulator, default_params, r_wave_gain
    sim = ECGSimulator()
    p = default_params(3, gain=r_wave_gain())
    hr = torch.tensor([55.0, 75.0, 110.0])
    x = sim.simulate(p, 10.0, hr, torch.full((3,), 0.01), torch.full((3,), 0.25),
                     gen=torch.Generator().manual_seed(0))
    assert x.shape == (3, 1000)
    for i, h in enumerate(hr.tolist()):
        est, _ = heart_rate(x[i].numpy())
        assert abs(est - h) < 0.08 * h


@needs_torch
def test_simulator_is_differentiable():
    from sensing_gen.physics.ecg_ode import ECGSimulator, default_params
    p = default_params(1)
    p.a.requires_grad_(True)
    y = ECGSimulator().simulate_beat(p, torch.tensor([0.9]))
    y.pow(2).sum().backward()
    assert torch.isfinite(p.a.grad).all() and p.a.grad.abs().sum() > 0


@needs_torch
def test_calibration_recovers_simulated_beat():
    from sensing_gen.physics.calibrate import fit_beats
    from sensing_gen.physics.ecg_ode import ECGSimulator, default_params, r_wave_gain
    p = default_params(2, gain=r_wave_gain())
    p.a[1, 4] *= -1.0  # inverted T wave
    rr = torch.tensor([0.8, 1.0])
    beats = ECGSimulator().simulate_beat(p, rr).numpy()
    _, r2, _ = fit_beats(beats, rr.numpy(), steps=300)
    assert (r2 > 0.95).all()


@needs_torch
def test_sensor_model_snr_and_determinism():
    from sensing_gen.physics.ecg_ode import ECGSimulator, default_params, r_wave_gain
    from sensing_gen.physics.sensor import ElectrodeSkinSensor, SensorConfig
    x = ECGSimulator().simulate(default_params(16, gain=r_wave_gain()), 10.0, torch.full((16,), 70.0),
                                torch.full((16,), 0.02), torch.full((16,), 0.25))
    s = ElectrodeSkinSensor(SensorConfig(p_contact_loss=0.0))
    y1 = s(x, torch.Generator().manual_seed(0), snr_db=torch.full((16,), 6.0))
    y2 = s(x, torch.Generator().manual_seed(0), snr_db=torch.full((16,), 6.0))
    assert torch.equal(y1, y2)
    meas = [snr_db(a, b) for a, b in zip(x.numpy(), y1.numpy())]
    assert 3.0 < np.median(meas) < 7.0  # gain modulation + quantisation shave a little off 6 dB


# ---------------------------------------------------------------- generative model
@needs_torch
def test_unet_and_diffusion_shapes():
    from sensing_gen.diffusion.ddpm import GaussianDiffusion
    from sensing_gen.diffusion.unet1d import UNet1D
    d = GaussianDiffusion(UNet1D(base=16, mults=(1, 2, 4)), T=50)
    x = torch.randn(4, 1, 1000)
    y = torch.tensor([0, 1, 0, 1])
    hr = torch.tensor([60.0, 70.0, 80.0, 90.0])
    out = d.model(x, torch.randint(0, 50, (4,)), y, hr)
    assert out.shape == x.shape
    loss = d.loss(x, y, hr)
    loss.backward()
    assert torch.isfinite(loss)
    s = d.ddim_sample(4, 1000, y, hr, steps=5, w=1.0)
    assert s.shape == (4, 1, 1000) and torch.isfinite(s).all()


@needs_torch
def test_q_sample_marginal_variance():
    from sensing_gen.diffusion.ddpm import GaussianDiffusion
    from sensing_gen.diffusion.unet1d import UNet1D
    d = GaussianDiffusion(UNet1D(base=16), T=100)
    x0 = torch.zeros(2000, 1, 8)
    t = torch.full((2000,), 99)
    xt = d.q_sample(x0, t, torch.randn_like(x0))
    assert abs(xt.var().item() - (1 - d.ab[99]).item()) < 0.05


# ---------------------------------------------------------------- PINN
@needs_torch
def test_pinn_runs_and_reduces_data_loss():
    from sensing_gen.physics.ecg_ode import default_params, r_wave_gain
    from sensing_gen.pinn.ecg_pinn import fit_pinn, phase_from_rpeaks, simulate_from_phase
    rp = [np.arange(-3, 10, 0.85), np.arange(-3, 10, 0.75)]
    fns = [lambda t, r=r: phase_from_rpeaks(r, t) for r in rp]
    z, _ = simulate_from_phase(default_params(2, gain=r_wave_gain()), fns, 10.0)
    y = z[:, ::5][:, :1000].numpy() + 0.05 * np.random.default_rng(0).standard_normal((2, 1000))
    peaks = [r[(r >= 0) & (r < 10)] for r in rp]
    res = fit_pinn(y.astype(np.float32), peaks, steps=60, diffsim_steps=40, lam=1.0)
    assert res.denoised.shape == (2, 1000)
    assert res.history[0] > 0 and np.isfinite(res.denoised).all()


# ---------------------------------------------------------------- multiphysics (numpy / scipy only)
def test_fv_solver_matches_thin_strip_analytic():
    from sensing_gen.multiphysics.electrode_skin_fv import (ElectrodeConfig, Layer, TissueModel,
                                                            contact_state, interface_admittance,
                                                            solve_impedance)
    tm = TissueModel(layers=[Layer("strip", 2e-4, 0.2, 1.0)])
    ec = ElectrodeConfig(kind="dry")
    f = 0.01
    z = solve_impedance(tm, ec, f)
    y = interface_admittance(ec, tm, contact_state(ec, tm, 5e3), 2 * math.pi * f)
    st = 0.2 * 2e-4 * ec.length
    z_an = 2 / (y * ec.width * ec.length) + (ec.spacing - ec.width) / st + 2 * ec.width / (3 * st)
    assert abs(abs(z) / abs(z_an) - 1) < 0.01


def test_fv_reciprocity_and_physics_trends():
    from sensing_gen.multiphysics.electrode_skin_fv import ElectrodeConfig, TissueModel, solve_impedance
    tm = TissueModel()
    dry, wet = ElectrodeConfig(kind="dry"), ElectrodeConfig(kind="wet")
    za = solve_impedance(tm, dry, 10.0, pressure=(3e3, 8e3))
    zb = solve_impedance(tm, dry, 10.0, pressure=(8e3, 3e3))
    assert abs(za - zb) / abs(za) < 1e-6                                   # reciprocity
    assert abs(solve_impedance(tm, wet, 10.0)) < abs(solve_impedance(tm, dry, 10.0))  # gel lowers Z
    z0 = abs(solve_impedance(tm, dry, 10.0))
    assert abs(solve_impedance(tm, dry, 10.0, lift_off=(0.4, 0.0))) > z0    # lift-off raises Z
    assert abs(solve_impedance(tm, dry, 1000.0)) < z0                       # capacitive roll-off


@needs_torch
def test_sensor_default_priors_match_multiphysics_check():
    from sensing_gen.physics.sensor import SensorConfig
    c = SensorConfig()
    assert tuple(c.kappa_range) == (0.0, 0.5) and tuple(c.ze_ratio_range) == (0.001, 0.05)


@needs_torch
def test_ddim_leading_dynamic_sampler_is_finite_and_bounded():
    from sensing_gen.diffusion.ddpm import GaussianDiffusion
    from sensing_gen.diffusion.unet1d import UNet1D
    d = GaussianDiffusion(UNet1D(base=16), T=1000)
    y, hr = torch.tensor([0, 1]), torch.tensor([60.0, 90.0])
    s = d.ddim_sample(2, 1000, y, hr, steps=10, w=1.0, spacing="leading", threshold="dynamic")
    assert s.shape == (2, 1, 1000) and torch.isfinite(s).all() and s.abs().max() <= 12.0 + 1e-4


def test_read_record_multi_frequency(tmp_path):
    from sensing_gen.data.wfdb_io import read_record_multi
    # signal A: 4 samples per frame, signal B: 1 sample per frame, 10 frames, format 16
    a = np.arange(40, dtype=np.int16)
    b = 1000 + np.arange(10, dtype=np.int16)
    frames = np.concatenate([a.reshape(10, 4), b.reshape(10, 1)], axis=1).astype("<i2")
    (tmp_path / "m.dat").write_bytes(frames.tobytes())
    (tmp_path / "m.hea").write_text("m 2 15.5 10\nm.dat 16x4 1(0)/mV 16 0 0 0 0 ECG\nm.dat 16 1(0)/mV 16 0 0 0 0 RESP\n")
    sigs, hdr = read_record_multi(tmp_path / "m")
    assert np.array_equal(sigs["ECG"][0], a) and sigs["ECG"][1] == 62.0
    assert np.array_equal(sigs["RESP"][0], b) and sigs["RESP"][1] == 15.5


def test_compare_comsol_roundtrip(tmp_path):
    from sensing_gen.multiphysics import compare_comsol as cc
    f = np.array([1.0, 10.0, 100.0])
    z = cc.python_reference(f)
    csv = tmp_path / "c.csv"
    csv.write_text("% freq (Hz), abs(ec.Z11) (Ohm)\n" + "\n".join(f"{a},{abs(b)}" for a, b in zip(f, z)))
    fr, mag, _ = cc.read_table(csv, current=False)
    assert np.allclose(fr, f) and np.allclose(mag, np.abs(z))
    md = tmp_path / "R.md"
    md.write_text(f"a\n{cc.MARK_BEGIN}\nold\n{cc.MARK_END}\nb\n")
    cc.update_readme(md, "new line")
    assert "new line" in md.read_text() and "old" not in md.read_text()
