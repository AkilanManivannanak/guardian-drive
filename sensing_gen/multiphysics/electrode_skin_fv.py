"""Electro-mechanical multiphysics model of a biopotential electrode on layered tissue.

Physics coupled here:

1. **Contact mechanics (quasi-static).** Normal pressure p on the electrode, set by seat/garment
   preload plus body motion, controls
     * dry electrodes: the real contact-area fraction alpha(p) = min(1, (p / p_sat)^m), a
       Greenwood-Williamson-type asperity-contact scaling;
     * wet (gel) electrodes: gel-layer thickness, by linear-elastic compression h = h0 (1 - p/E_gel);
     * both: stratum-corneum thickness, by linear-elastic compression h = h0 (1 - p/E_sc);
     * lift-off: a fraction of the electrode width that loses contact entirely.
2. **Electro-quasi-statics (EQS)** in a 2-D cross-section of layered tissue
   (viable epidermis/dermis, subcutaneous fat, muscle), with complex conductivity
   sigma* = sigma + j omega eps0 eps_r:

        div( sigma*(x, z) grad phi ) = 0

   discretised with a cell-centred finite-volume scheme on a tensor-product grid that is
   geometrically refined toward the skin surface. Harmonic-mean face conductances, insulating outer
   boundaries, and a thin-layer (lumped) boundary condition for the stratum corneum + gel / air-gap
   contact under each electrode. Solved with a sparse direct solver.

Outputs: two-electrode impedance spectra, and the sensitivity of electrode impedance to motion. That
sensitivity is the kappa parameter of the lumped sensor model in sensing_gen/physics/sensor.py, so
this model provides a physics-based check of that model's prior range.

Material values are representative low-frequency literature values for skin layers (e.g. Gabriel
et al. 1996, Yamamoto & Yamamoto 1976) and should be read as nominal, not patient-specific. The
model is 2-D (per unit electrode length), so absolute impedances are approximate; relative changes,
which is what the sensor model needs, are the robust output.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EPS0 = 8.854e-12


@dataclass
class Layer:
    name: str
    thickness: float      # m
    sigma: float          # S/m
    eps_r: float


@dataclass
class TissueModel:
    layers: list = field(default_factory=lambda: [
        Layer("epidermis+dermis", 1.5e-3, 0.20, 1.0e4),
        Layer("fat", 4.0e-3, 0.04, 1.0e4),
        Layer("muscle", 24.5e-3, 0.35, 1.0e5),
    ])
    # stratum corneum (thin layer, lumped)
    sc_thickness: float = 20e-6
    sc_sigma: float = 2e-6         # dry (non-hydrated) stratum corneum
    sc_hydration_gain: float = 10.0  # conductivity gain when hydrated by electrolyte gel
    sc_eps_r: float = 1.0e3
    sc_modulus: float = 2.0e6      # Pa
    width: float = 0.06            # m, lateral extent of the cross-section

    @property
    def depth(self) -> float:
        return sum(l.thickness for l in self.layers)


@dataclass
class ElectrodeConfig:
    kind: str = "dry"              # "dry" or "wet"
    width: float = 10e-3           # m
    length: float = 10e-3          # m, out-of-plane length used to convert 2-D -> ohms
    spacing: float = 30e-3         # centre-to-centre
    # wet electrode gel
    gel_thickness: float = 1.0e-3
    gel_sigma: float = 1.0
    gel_eps_r: float = 80.0
    gel_modulus: float = 30e3      # Pa
    # dry electrode asperity contact
    p_sat: float = 20e3            # Pa, pressure at which contact becomes complete
    contact_exponent: float = 0.6
    air_gap: float = 5e-6          # m, mean gap across non-contacting area
    # contact-interface double layer (wet and dry), area-specific
    interface_r: float = 2.0e-3    # ohm m^2 (charge-transfer resistance, polarisable electrode)
    interface_c: float = 0.5       # F/m^2 (double-layer capacitance)


def make_grid(tm: TissueModel, nx: int = 240, dz0: float = 10e-6, ratio: float = 1.12):
    """Uniform in x, geometric in z (fine near the skin surface)."""
    x_edges = np.linspace(-tm.width / 2, tm.width / 2, nx + 1)
    dz, z = [], 0.0
    d = dz0
    while z < tm.depth - 1e-12:
        d_eff = min(d, tm.depth - z)
        dz.append(d_eff)
        z += d_eff
        d *= ratio
    z_edges = np.concatenate([[0.0], np.cumsum(dz)])
    return x_edges, z_edges


def _material(tm: TissueModel, zc: np.ndarray, omega: float) -> np.ndarray:
    tops = np.cumsum([0.0] + [l.thickness for l in tm.layers])
    s = np.empty_like(zc, dtype=complex)
    for l, t0, t1 in zip(tm.layers, tops[:-1], tops[1:]):
        m = (zc >= t0) & (zc < t1 + 1e-15)
        s[m] = l.sigma + 1j * omega * EPS0 * l.eps_r
    return s


def contact_state(ec: ElectrodeConfig, tm: TissueModel, pressure: float, lift_off: float = 0.0) -> dict:
    """Contact mechanics: geometric state of the electrode-skin contact under normal pressure."""
    p = max(pressure, 0.0)
    h_sc = tm.sc_thickness * max(0.2, 1 - p / tm.sc_modulus)
    if ec.kind == "wet":
        alpha = 1.0
        h_gel = ec.gel_thickness * max(0.2, 1 - p / ec.gel_modulus)
    else:
        alpha = float(min(1.0, (p / ec.p_sat) ** ec.contact_exponent)) if p > 0 else 0.0
        h_gel = 0.0
    return {"h_sc": h_sc, "alpha": max(alpha, 1e-3), "h_gel": h_gel,
            "contact_width": ec.width * (1 - np.clip(lift_off, 0, 0.95))}


def interface_admittance(ec: ElectrodeConfig, tm: TissueModel, cs: dict, omega: float) -> complex:
    """Area-specific admittance [S/m^2] of: electrode double layer -> (gel) -> stratum corneum,
    in parallel with an air-gap capacitance over the non-contacting fraction (dry electrodes)."""
    z_dl = 1.0 / (1.0 / ec.interface_r + 1j * omega * ec.interface_c)          # ohm m^2
    sc_sigma = tm.sc_sigma * (tm.sc_hydration_gain if ec.kind == "wet" else 1.0)
    y_sc = (sc_sigma + 1j * omega * EPS0 * tm.sc_eps_r) / cs["h_sc"]              # S/m^2
    z = z_dl + 1.0 / y_sc
    if cs["h_gel"] > 0:
        z += cs["h_gel"] / (ec.gel_sigma + 1j * omega * EPS0 * ec.gel_eps_r)
    y_contact = cs["alpha"] / z
    y_gap = (1 - cs["alpha"]) * (1j * omega * EPS0) / ec.air_gap if cs["alpha"] < 1 else 0.0
    return y_contact + y_gap


def solve_impedance(tm: TissueModel, ec: ElectrodeConfig, freq: float, pressure: tuple = (5e3, 5e3),
                    lift_off: tuple = (0.0, 0.0), grid=None, return_field: bool = False):
    """Two-electrode impedance Z (ohm, converted from 2-D using ec.length) at frequency freq.

    pressure / lift_off are (electrode A, electrode B) so a single moving electrode can be modelled.
    """
    omega = 2 * np.pi * freq
    x_e, z_e = grid if grid is not None else make_grid(tm)
    xc, zc = 0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (z_e[1:] + z_e[:-1])
    dx, dz = np.diff(x_e), np.diff(z_e)
    nx, nz = len(xc), len(zc)
    sig_z = _material(tm, zc, omega)                      # layered: depends on z only
    S = np.broadcast_to(sig_z[None, :], (nx, nz))
    idx = np.arange(nx * nz).reshape(nx, nz)

    rows, cols, vals = [], [], []
    diag = np.zeros(nx * nz, dtype=complex)
    # x-faces
    R = (dx[:-1, None] / 2) / S[:-1] + (dx[1:, None] / 2) / S[1:]
    G = dz[None, :] / R
    a, b = idx[:-1].ravel(), idx[1:].ravel()
    g = G.ravel()
    rows += [a, b]; cols += [b, a]; vals += [-g, -g]
    np.add.at(diag, a, g); np.add.at(diag, b, g)
    # z-faces
    R = (dz[None, :-1] / 2) / S[:, :-1] + (dz[None, 1:] / 2) / S[:, 1:]
    G = dx[:, None] / R
    a, b = idx[:, :-1].ravel(), idx[:, 1:].ravel()
    g = G.ravel()
    rows += [a, b]; cols += [b, a]; vals += [-g, -g]
    np.add.at(diag, a, g); np.add.at(diag, b, g)

    # electrodes on the top surface (thin-layer boundary condition)
    rhs = np.zeros(nx * nz, dtype=complex)
    centres = (-ec.spacing / 2, ec.spacing / 2)
    volts = (1.0, 0.0)
    g_top_all = []
    for e in range(2):
        cs = contact_state(ec, tm, pressure[e], lift_off[e])
        y_if = interface_admittance(ec, tm, cs, omega)
        half = cs["contact_width"] / 2
        lo, hi = centres[e] - half, centres[e] + half
        frac = np.clip((np.minimum(x_e[1:], hi) - np.maximum(x_e[:-1], lo)) / dx, 0, 1)  # covered fraction
        cells = np.nonzero(frac > 0)[0]
        # series: interface (per area) + half top cell of tissue
        g_cell = 1.0 / (1.0 / (y_if * frac[cells] * dx[cells]) + (dz[0] / 2) / (sig_z[0] * frac[cells] * dx[cells]))
        k = idx[cells, 0]
        diag[k] += g_cell
        rhs[k] += g_cell * volts[e]
        g_top_all.append((k, g_cell, volts[e]))
    A = sp.csr_matrix((np.concatenate(vals + [diag]),
                       (np.concatenate(rows + [np.arange(nx * nz)]), np.concatenate(cols + [np.arange(nx * nz)]))),
                      shape=(nx * nz, nx * nz))
    phi = spla.spsolve(A.tocsc(), rhs)
    kA, gA, vA = g_top_all[0]
    I = np.sum(gA * (vA - phi[kA]))           # A per metre of electrode length
    Z2d = (volts[0] - volts[1]) / I           # ohm * m
    Z = Z2d / ec.length
    if return_field:
        return Z, phi.reshape(nx, nz), (xc, zc)
    return Z


def impedance_spectrum(tm: TissueModel, ec: ElectrodeConfig, freqs, **kw) -> np.ndarray:
    grid = make_grid(tm)
    return np.array([solve_impedance(tm, ec, f, grid=grid, **kw) for f in freqs])


def motion_sensitivity(tm: TissueModel, ec: ElectrodeConfig, p0: float = 5e3, dp_frac: float = 0.3,
                       lift: float = 0.2, freq: float = 10.0) -> dict:
    """Relative impedance change of the two-electrode pair when ONE electrode is perturbed.

    kappa_pressure: |Z(p0 (1 - dp)) - Z(p0)| / |Z(p0)| / dp   (relative change per relative unloading)
    kappa_liftoff:  |Z(lift) - Z(0)| / |Z(0)| / lift
    mismatch_ratio: |Z_A - Z_B| / |Z| for the perturbed electrode, which sets mains conversion.
    """
    grid = make_grid(tm)
    z0 = solve_impedance(tm, ec, freq, pressure=(p0, p0), grid=grid)
    zp = solve_impedance(tm, ec, freq, pressure=(p0 * (1 - dp_frac), p0), grid=grid)
    zl = solve_impedance(tm, ec, freq, pressure=(p0, p0), lift_off=(lift, 0.0), grid=grid)
    return {
        "Z0_ohm": abs(z0),
        "kappa_pressure": abs(zp - z0) / abs(z0) / dp_frac,
        "kappa_liftoff": abs(zl - z0) / abs(z0) / lift,
        "dZ_pressure_ohm": abs(zp - z0),
        "dZ_liftoff_ohm": abs(zl - z0),
    }


def with_changes(obj, **kw):
    return replace(obj, **kw)
