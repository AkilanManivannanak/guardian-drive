"""Export the trained surrogate for the in-browser electrode designer (docs/explorer.html).

Weights are stored as int8 with one float32 scale per output row (median |Z| error 0.032 % vs 0.028 %
in float32 on fresh FE samples), base64-encoded in docs/surrogate_web.json together with the parameter
ranges and a few FE-solved presets that the page plots as ground-truth points.

    python -m sensing_gen.surrogate.export_web
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from .dataset import simulate_one
from .space import FREQS, HI, LO, LOG, NAMES

PRESETS = {
    "Gel electrode, nominal": {},
    "Small electrode (6 mm)": {"w_el_mm": 6.0},
    "30 % lift-off": {"lift": 0.3},
    "Drier skin": {"sig_sc": 6e-6},
    "Thick fat layer (9 mm)": {"t_fat_mm": 9.0},
}
DEFAULTS = {"t_derm_mm": 1.5, "t_fat_mm": 4.0, "sig_derm": 0.20, "sig_fat": 0.04, "sig_mus": 0.35,
            "sig_sc": 2e-5, "eps_sc": 1000.0, "w_el_mm": 10.0, "s_el_mm": 30.0, "lift": 0.0}


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _layer(w: np.ndarray, b: np.ndarray) -> dict:
    s = np.abs(w).max(axis=1, keepdims=True) / 127 + 1e-12
    q = np.round(w / s).clip(-127, 127).astype(np.int8)
    return {"rows": int(w.shape[0]), "cols": int(w.shape[1]), "q": _b64(q), "s": _b64(s.astype(np.float32).ravel()),
            "b": _b64(b.astype(np.float32))}


def export(state: dict, cfg: dict, out: Path) -> dict:
    st = {k: np.asarray(v, dtype=np.float32) for k, v in state.items()}
    members = []
    for k in range(cfg["k"]):
        layers = [_layer(st[f"members.{k}.body.{j}.weight"], st[f"members.{k}.body.{j}.bias"]) for j in (0, 2, 4)]
        members.append({"layers": layers, "mu": _layer(st[f"members.{k}.mu.weight"], st[f"members.{k}.mu.bias"]),
                        "lv": _layer(st[f"members.{k}.logvar.weight"], st[f"members.{k}.logvar.bias"])})
    presets = {}
    for name, p in PRESETS.items():
        th = np.array([{**DEFAULTS, **p}[n] for n in NAMES])
        z = simulate_one(th)
        presets[name] = {"params": {**DEFAULTS, **p}, "fe_abs": np.abs(z).round(3).tolist(),
                         "fe_phase_deg": np.angle(z, deg=True).round(4).tolist()}
    doc = {"names": NAMES, "lo": LO.tolist(), "hi": HI.tolist(), "log": LOG.tolist(), "freqs": FREQS.tolist(),
           "y_mean": st["y_mean"].tolist(), "y_std": st["y_std"].tolist(), "defaults": DEFAULTS,
           "members": members, "presets": presets,
           "note": "int8 per-row quantised deep ensemble; prior = analytic contact impedance (sensing_gen/surrogate/space.py)"}
    out.write_text(json.dumps(doc, separators=(",", ":")))
    return doc


def main():
    from .model import load_surrogate
    root = Path(__file__).resolve().parents[2]
    m = load_surrogate(root / "sensing_gen" / "weights" / "electrode_surrogate.pt")
    state = {k: v.cpu().numpy() for k, v in m.state_dict().items()}
    export(state, m.cfg, root / "docs" / "surrogate_web.json")
    print("wrote docs/surrogate_web.json")


if __name__ == "__main__":
    main()
