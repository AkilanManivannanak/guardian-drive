"""Independent finite-element (P1 triangles) solver for the electrode-skin problem.

Used to cross-check the finite-volume solver in `electrode_skin_fv.py` with a different
discretisation. It solves the same electro-quasi-static problem

    div( sigma*(z) grad phi ) = 0      in layered tissue (complex conductivity)
    -sigma* d(phi)/dn = Y (phi - V_e)  on each electrode (contact-impedance / Robin condition)
    d(phi)/dn = 0                      elsewhere (insulating),

with the Galerkin weak form on a triangulated, surface-graded mesh. Here
Y = (sigma_sc + j omega eps0 eps_sc) / t_sc is the stratum-corneum layer admittance per area, which is
what COMSOL's "Contact Impedance" boundary condition implements. Only the "COMSOL-equivalent"
wet-electrode configuration is supported (no double layer, no gel, no pressure dependence).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .electrode_skin_fv import EPS0, ElectrodeConfig, TissueModel


def _nodes_x(width: float, centres, half_widths, h: float) -> np.ndarray:
    x = np.arange(-width / 2, width / 2 + 1e-12, h)
    edges = [c + s * hw for c, hw in zip(centres, half_widths) for s in (-1, 1)]
    x = np.unique(np.round(np.concatenate([x, edges]), 12))
    return x


def _nodes_z(depth: float, layer_tops, dz0: float, ratio: float) -> np.ndarray:
    z, d, zs = 0.0, dz0, [0.0]
    while z < depth - 1e-12:
        z = min(z + d, depth)
        zs.append(z)
        d *= ratio
    zs = np.unique(np.round(np.concatenate([zs, layer_tops]), 12))
    return zs[zs <= depth + 1e-15]


def fem_impedance(freq: float, tm: TissueModel | None = None, ec: ElectrodeConfig | None = None,
                  lift: float = 0.0, h: float = 0.25e-3, dz0: float = 10e-6, ratio: float = 1.12) -> complex:
    """Two-electrode impedance (ohm, for an electrode of length ec.length) by P1 finite elements."""
    tm = tm or TissueModel()
    ec = ec or ElectrodeConfig(kind="wet")
    omega = 2 * np.pi * freq
    centres = (-ec.spacing / 2, ec.spacing / 2)
    half = (ec.width * (1 - lift) / 2, ec.width / 2)
    tops = np.cumsum([0.0] + [l.thickness for l in tm.layers])
    xs = _nodes_x(tm.width, centres, half, h)
    zs = _nodes_z(tm.depth, tops, dz0, ratio)
    nx, nz = len(xs), len(zs)
    idx = np.arange(nx * nz).reshape(nx, nz)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    P = np.stack([X.ravel(), Z.ravel()], 1)

    # two triangles per rectangle
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    T = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])

    # element conductivity from the centroid depth
    zc = P[T, 1].mean(1)
    sig = np.empty(len(T), dtype=complex)
    for l, t0, t1 in zip(tm.layers, tops[:-1], tops[1:]):
        m = (zc >= t0) & (zc < t1 + 1e-15)
        sig[m] = l.sigma + 1j * omega * EPS0 * l.eps_r

    # P1 stiffness: K_e = sigma * area * G G^T
    p0, p1, p2 = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    area = 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
    bcoef = np.stack([p1[:, 1] - p2[:, 1], p2[:, 1] - p0[:, 1], p0[:, 1] - p1[:, 1]], 1)
    ccoef = np.stack([p2[:, 0] - p1[:, 0], p0[:, 0] - p2[:, 0], p1[:, 0] - p0[:, 0]], 1)
    Ke = (bcoef[:, :, None] * bcoef[:, None, :] + ccoef[:, :, None] * ccoef[:, None, :]) / (4 * area[:, None, None])
    Ke = Ke * sig[:, None, None]
    rows = np.repeat(T, 3, axis=1).ravel()
    cols = np.tile(T, (1, 3)).ravel()
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(nx * nz, nx * nz)).tocsr()

    # Robin (contact impedance) on the electrode segments of the top edge
    sc_sigma = tm.sc_sigma * tm.sc_hydration_gain
    Y = (sc_sigma + 1j * omega * EPS0 * tm.sc_eps_r) / tm.sc_thickness
    top = idx[:, 0]
    rhs = np.zeros(nx * nz, dtype=complex)
    rr, cc, vv = [], [], []
    seg_A = []
    for e, (cx, hw, V) in enumerate(zip(centres, half, (1.0, 0.0))):
        for i in range(nx - 1):
            xm = 0.5 * (xs[i] + xs[i + 1])
            if cx - hw - 1e-12 <= xm <= cx + hw + 1e-12:
                L = xs[i + 1] - xs[i]
                n0, n1 = top[i], top[i + 1]
                for (p, q, w) in ((n0, n0, 2), (n1, n1, 2), (n0, n1, 1), (n1, n0, 1)):
                    rr.append(p); cc.append(q); vv.append(Y * L * w / 6)
                rhs[n0] += Y * L * V / 2
                rhs[n1] += Y * L * V / 2
                if e == 0:
                    seg_A.append((n0, n1, L))
    K = K + sp.coo_matrix((vv, (rr, cc)), shape=K.shape).tocsr()
    phi = spla.spsolve(K.tocsc(), rhs)
    I = sum(Y * L * (1.0 - 0.5 * (phi[n0] + phi[n1])) for n0, n1, L in seg_A)  # A per metre
    return (1.0 / I) / ec.length
