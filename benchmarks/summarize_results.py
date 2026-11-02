"""Validate and summarize the checked-in combined-training benchmark artifact."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "outputs" / "combined_training_results.json"
REQUIRED_FIELDS = {
    "model",
    "n_real",
    "n_synthetic",
    "total",
    "best_loss",
    "epochs",
    "training_progression",
}


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def load_results() -> Dict[str, Any]:
    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(f"Benchmark result artifact is missing: {RESULTS_PATH}")
    with RESULTS_PATH.open(encoding="utf-8") as stream:
        results = json.load(stream)
    if not isinstance(results, dict):
        raise ValueError("Combined training benchmark must be a JSON object")

    missing = REQUIRED_FIELDS.difference(results)
    if missing:
        raise ValueError(f"Benchmark artifact is missing fields: {sorted(missing)}")
    for field in ("n_real", "n_synthetic", "total", "epochs"):
        value = results[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    if results["total"] != results["n_real"] + results["n_synthetic"]:
        raise ValueError("total must equal n_real + n_synthetic")
    if results["epochs"] < 1:
        raise ValueError("epochs must be at least 1")
    if not isinstance(results["model"], str) or not results["model"].strip():
        raise ValueError("model must be a non-empty string")
    if _finite_number(results["best_loss"], "best_loss") < 0:
        raise ValueError("best_loss must be non-negative")

    progression = results["training_progression"]
    if not isinstance(progression, dict) or not progression:
        raise ValueError("training_progression must be a non-empty object")
    for epoch, loss in progression.items():
        if not isinstance(epoch, str) or not epoch.startswith("epoch_"):
            raise ValueError(f"Invalid training_progression key: {epoch!r}")
        if _finite_number(loss, f"training_progression.{epoch}") < 0:
            raise ValueError(f"training_progression.{epoch} must be non-negative")
    return results


def main() -> None:
    results = load_results()
    print(f"Benchmark: {results['model']}")
    print(
        "Samples: "
        f"{results['n_real']} real + {results['n_synthetic']} synthetic "
        f"= {results['total']} total"
    )
    print(f"Epochs: {results['epochs']}; best loss: {results['best_loss']:.6g}")
    print(f"Training checkpoints recorded: {len(results['training_progression'])}")
    print("Benchmark artifact validation: OK")


if __name__ == "__main__":
    main()
