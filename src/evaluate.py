"""Metrics and experiment logging.

Metrics (all anchors weighted equally):
- mae_rate, rmse_rate: error of the predicted consumption rate, in %SoC/km
- mae_dte_km: error of the implied distance-to-empty, in km

DTE = usable SoC / rate is undefined for rates <= 0 (possible for TripSoFar after a downhill
start). Predicted rates are therefore floored at RATE_FLOOR before dividing; how often that
happens is reported as `share_floored`.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LOG_PATH = Path(__file__).resolve().parents[1] / "experiments" / "log.csv"
LOG_FIELDS = ["timestamp", "run_id", "phase", "model", "split", "config_json",
              "mae_rate", "rmse_rate", "mae_dte_km", "notes"]
# The lowest actual 5 km rate in the data is 0.10 %SoC/km (1st percentile 0.24), so a floor of
# 0.05 only catches predictions that are implausible for a 5 km horizon.
RATE_FLOOR = 0.05


def implied_dte(usable_soc_pct: np.ndarray, rate: np.ndarray, floor: float = RATE_FLOOR) -> np.ndarray:
    """Distance-to-empty in km from usable SoC (%-points) and a consumption rate (%SoC/km)."""
    return np.asarray(usable_soc_pct, float) / np.maximum(np.asarray(rate, float), floor)


def metrics(rate_true: np.ndarray, rate_pred: np.ndarray, usable_soc_pct: np.ndarray) -> dict[str, float]:
    """Compute rate MAE/RMSE, DTE MAE (km) and the share of predictions hitting the rate floor."""
    rate_true, rate_pred = np.asarray(rate_true, float), np.asarray(rate_pred, float)
    err = rate_pred - rate_true
    dte_err = implied_dte(usable_soc_pct, rate_pred) - implied_dte(usable_soc_pct, rate_true)
    return {
        "mae_rate": float(np.mean(np.abs(err))),
        "rmse_rate": float(np.sqrt(np.mean(err ** 2))),
        "mae_dte_km": float(np.mean(np.abs(dte_err))),
        "share_floored": float(np.mean(rate_pred < RATE_FLOOR)),
    }


def log_experiment(run_id: str, phase: str, model: str, split: str, config: dict,
                   result: dict[str, float], notes: str = "", path: Path = LOG_PATH) -> None:
    """Append one run (config + metrics) as a row of experiments/log.csv."""
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_id": run_id, "phase": phase, "model": model, "split": split,
            "config_json": json.dumps(config, sort_keys=True),
            "mae_rate": f"{result['mae_rate']:.4f}", "rmse_rate": f"{result['rmse_rate']:.4f}",
            "mae_dte_km": f"{result['mae_dte_km']:.3f}", "notes": notes,
        })
