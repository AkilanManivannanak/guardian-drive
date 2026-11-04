# Self-recorded wearable ECG: recording protocol

Purpose: test the physics sensor model and the PTB-XL classifiers on a real consumer device, under
conditions where motion artifacts appear. Analysis: `python -m sensing_gen.experiments.run_all wearable`
(`sensing_gen/experiments/run_wearable.py`).

## Safety and privacy

* **Never record while driving.** The car condition is recorded as a passenger only.
* Raw recordings are personal health data. They stay in `data/sensing_gen/wearable/`, which git ignores.
  Only aggregate statistics (window counts, quality, spectra, alarm rates) are written to `outputs/`.
* The analysis assumes the person recording has no known cardiac condition, so every "abnormal" output is
  counted as a false alarm. If that is not true, do not use this protocol to draw conclusions.

## Conditions

| folder name | what to do | Apple Watch (30-s recordings) | chest strap / board (continuous) |
|---|---|---|---|
| `still` | seated, both forearms resting on a table, breathe normally | 8 | 4 min |
| `standing` (optional) | standing still | 4 | 2 min |
| `walking` | slow walk indoors; for the watch, keep the finger on the crown | 8 | 4 min |
| `arm_motion` | seated; move the arm wearing the sensor (small gestures, as when talking) | 8 | 4 min |
| `car_passenger` | passenger seat on normal roads, seatbelt on | 8 | 6 min |

Record the conditions in blocks and note the start and end time of each block. Leave 1 minute between
blocks. Keep the same wrist / strap position for the whole session; wet the strap electrodes before
starting (Polar H10).

## Devices

**Apple Watch (ECG app, lead I, 512 Hz).** After the session: iPhone Health app → profile picture →
*Export All Health Data* → send `export.zip` to the Mac and unzip it. The recordings are in
`apple_health_export/electrocardiograms/`. Write the time blocks to `plan.csv`:

```
condition,start,end
still,2026-11-05 09:00,2026-11-05 09:15
walking,2026-11-05 09:20,2026-11-05 09:40
arm_motion,2026-11-05 09:45,2026-11-05 10:00
car_passenger,2026-11-05 17:30,2026-11-05 18:00
```

then sort them into condition folders (reads each file's "Recorded Date"; nothing else from the header is used):

```bash
python -m sensing_gen.data.wearable_io ~/Downloads/apple_health_export/electrocardiograms plan.csv
```

**Polar H10 (130 Hz).** Record with the *Polar Sensor Logger* app (ECG stream on), one file per block,
and copy each `*_ECG.txt` into the condition folder.

**AD8232 board + Arduino.** Log one value per line over serial, save one file per block, and put the
sample rate in `fs.txt` inside each condition folder (for example `250`).

## Run

```bash
SG_PROFILE=full python -m sensing_gen.experiments.run_all wearable
```

Outputs: `outputs/sensing_gen/wearable_results.json`, `fig_wearable.png`, and a section in `RESULTS.md`.
