#!/usr/bin/env bash
# Reproduce every sensing_gen number from scratch.
#   bash scripts/run_sensing_gen.sh          # download -> tests -> smoke run -> full run
#   bash scripts/run_sensing_gen.sh smoke    # stop after the smoke run
#   bash scripts/run_sensing_gen.sh full     # skip the smoke run
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-all}"
LOG=outputs/sensing_gen/logs
mkdir -p "$LOG"
export PYTHONUNBUFFERED=1
PY="${PYTHON:-python}"
KEEPAWAKE=""
command -v caffeinate >/dev/null 2>&1 && KEEPAWAKE="caffeinate -is"

"$PY" - <<'PY'
import importlib, subprocess, sys
need = [m for m in ("torch", "numpy", "scipy", "matplotlib", "pytest", "certifi")
        if importlib.util.find_spec(m) is None]
if need:
    print("[setup] installing", need)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *need])
import torch
print(f"[setup] torch {torch.__version__} | mps={torch.backends.mps.is_available()} | cuda={torch.cuda.is_available()}")
PY

echo "[1/4] downloading PTB-XL subset + MIT-BIH NSTDB from PhysioNet"
$KEEPAWAKE "$PY" -m sensing_gen.data.download 2>&1 | tee "$LOG/download.log"

echo "[2/4] unit tests"
"$PY" -m pytest -q tests/test_sensing_gen.py 2>&1 | tee "$LOG/tests.log"

if [ "$MODE" != "full" ]; then
  echo "[3/4] smoke run (tiny settings, catches errors before the long run)"
  SG_PROFILE=smoke $KEEPAWAKE "$PY" -m sensing_gen.experiments.run_all 2>&1 | tee "$LOG/smoke.log"
  [ "$MODE" = "smoke" ] && { echo "smoke run finished"; exit 0; }
fi

echo "[4/4] full run (~2-3 h on an Apple-silicon laptop)"
SG_PROFILE=full $KEEPAWAKE "$PY" -m sensing_gen.experiments.run_all 2>&1 | tee "$LOG/full.log"
echo "done -> outputs/sensing_gen/RESULTS.md"
