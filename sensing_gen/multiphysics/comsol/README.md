# COMSOL cross-check of the electrode–skin model

The goal is to rebuild the wet-electrode case of `electrode_skin_fv.py` in COMSOL Multiphysics and compare the two
with one command. Budget: one evening on the COMSOL trial (AC/DC Module).

> **License terms first.** A COMSOL *evaluation* (trial) license allows experimental and trial use only. It does not
> allow academic, commercial, professional or production use, and **results obtained with it may not be disseminated**.
> With a trial license, run the comparison privately and do **not** use `--update-readme` or commit
> `comsol_comparison.json` / `fig_comsol.png`. Publish COMSOL results only with a full (e.g. university) license, or
> with written permission from COMSOL.

## 1. Build the model (about 45 minutes)

1. **New model:** *2D* → *AC/DC* → *Electric Currents (ec)* → *Frequency Domain* study.
2. **Parameters:** *Global Definitions → Parameters → Load from File* → `parameters.txt` (this folder).
3. **Out-of-plane thickness:** *Electric Currents → Out-of-plane thickness* `d = d_oop` (10 mm). COMSOL then reports
   ohms for a 10 mm-long electrode, which is what the Python model reports.
4. **Geometry:** three stacked rectangles, all of width `W`, centred on x = 0, with the skin surface at y = 0:
   - dermis: height `t_derm`, y from −`t_derm` to 0
   - fat: height `t_fat`, directly below the dermis
   - muscle: height `t_mus`, directly below the fat

   Add four points on the top edge, at x = −`x_el` ± `w_el`/2 and x = +`x_el` ± `w_el`/2, so the top edge splits
   into two electrode boundaries.
5. **Materials:** three blank materials with *Electrical conductivity* and *Relative permittivity* set to
   (`sig_derm`, `eps_derm`), (`sig_fat`, `eps_fat`) and (`sig_mus`, `eps_mus`).
6. **Physics:**
   - *Terminal* (type: Voltage, `V0`) on the left electrode boundary.
   - *Ground* on the right electrode boundary.
   - *Contact Impedance* on both electrode boundaries: specify *Thickness and impedance*, with layer thickness `t_sc`,
     conductivity `sig_sc`, relative permittivity `eps_sc`.
   - Every other boundary stays *Electric Insulation* (the default).
7. **Mesh:** *Physics-controlled, Extra fine*, plus a *Boundary Layers* or *Size* node refined to 20 µm at the four
   electrode edge points.
8. **Study:** *Frequency Domain*, frequencies `10^range(0,0.2,4)`, which gives 21 points from 1 Hz to 10 kHz.

## 2. Export (about 5 minutes)

*Results → Derived Values → Global Evaluation*, with expressions `abs(ec.Z11)` and `arg(ec.Z11)*180/pi`.
If `ec.Z11` is not offered, use the terminal current `abs(ec.I0_1)` and pass `--current` below.
Evaluate, then *Export → Table* → `comsol_wet_sweep.csv`.

For lift-off, narrow the **left** electrode to 8 mm (move its two points inward by 1 mm each), re-run the study at
10 Hz only, and export `comsol_wet_liftoff8mm.csv`.

## 3. Compare (1 minute)

```bash
python -m sensing_gen.multiphysics.compare_comsol \
    --csv comsol_wet_sweep.csv --liftoff-csv comsol_wet_liftoff8mm.csv \
    --comsol-version "6.2" --update-readme
```

This writes `outputs/sensing_gen/comsol_comparison.json` and `fig_comsol.png`. With `--update-readme`, it also replaces
the "COMSOL cross-check" line in both READMEs with the measured agreement.

**Python targets** for this exact configuration (no electrode double layer, no gel, zero pressure):

| frequency | abs(Z) | phase |
|---|---|---|
| 1 Hz | 21.46 kΩ | −0.2° |
| 10 Hz | 21.45 kΩ | −1.5° |
| 100 Hz | 20.68 kΩ | −14.5° |
| 1 kHz | 7.40 kΩ | −59.6° |
| 10 kHz | 1.71 kΩ | −31.5° |
| 10 Hz, left electrode 8 mm | +11.9 % | — |

Agreement within about 5 % means the two solvers implement the same physics. Larger gaps usually come from the
mesh at the electrode edges or from the out-of-plane thickness. Add COMSOL to the resume only after this check
passes.
