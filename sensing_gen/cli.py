"""Command line: `sensing-gen augment | generate | impedance` (or `python -m sensing_gen.cli ...`)."""
from __future__ import annotations

import argparse
import json

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sensing-gen", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("augment", help="apply the physics sensor model to ECG in a .npy file (mV)")
    a.add_argument("inp"); a.add_argument("out")
    a.add_argument("--fs", type=float, default=100.0); a.add_argument("--seed", type=int, default=0)
    a.add_argument("--copies", type=int, default=1, help="noisy copies per input record")
    g = sub.add_parser("generate", help="sample synthetic 10-s ECG to a .npz file (x in mV, y labels)")
    g.add_argument("out"); g.add_argument("--n", type=int, default=64)
    g.add_argument("--label", default="norm", choices=["norm", "abnormal"])
    g.add_argument("--hr", type=float, default=70.0); g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="cpu"); g.add_argument("--weights", default=None)
    z = sub.add_parser("impedance", help="electrode impedance spectrum from the surrogate (or --exact FE)")
    z.add_argument("--exact", action="store_true")
    for k, v in {"t_derm_mm": 1.5, "t_fat_mm": 4.0, "sig_sc": 2e-5, "eps_sc": 1000.0,
                 "w_el_mm": 10.0, "s_el_mm": 30.0, "lift": 0.0}.items():
        z.add_argument("--" + k.replace("_", "-"), dest=k, type=float, default=v)
    args = ap.parse_args(argv)

    from . import api
    if args.cmd == "augment":
        x = np.load(args.inp)
        aug = api.SensorAugmenter(seed=args.seed)
        y = np.concatenate([aug(x, fs=args.fs) for _ in range(args.copies)])
        np.save(args.out, y.astype(np.float32))
        print(f"wrote {args.out}: {y.shape} (mV, 100 Hz)")
    elif args.cmd == "generate":
        gen = api.ECGGenerator.from_pretrained(args.weights, device=args.device)
        x, y = gen.sample(args.n, label=args.label, hr_bpm=args.hr, seed=args.seed)
        np.savez_compressed(args.out, x=x.astype(np.float32), y=y, fs=100.0)
        print(f"wrote {args.out}: {x.shape} (mV, 100 Hz), label={args.label}, hr={args.hr:g} bpm")
    else:
        p = {k: getattr(args, k) for k in ("t_derm_mm", "t_fat_mm", "sig_sc", "eps_sc", "w_el_mm", "s_el_mm", "lift")}
        sur = api.ElectrodeSurrogate() if args.exact else api.ElectrodeSurrogate.from_pretrained()
        zz, sd = sur.impedance(exact=args.exact, **p)
        f = api.ElectrodeSurrogate.frequencies()
        rows = [{"freq_hz": float(fi), "abs_z_ohm": float(abs(v)), "phase_deg": float(np.angle(v, deg=True)),
                 **({"sigma_log10": float(s)} if sd is not None else {})} for fi, v, s in
                zip(f, zz, sd if sd is not None else [None] * len(f))]
        print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
