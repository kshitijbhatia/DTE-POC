"""Tests for splits, targets, baselines and metrics (Phase 2) on synthetic data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.clean import add_derived_columns
from src.data.splits import make_splits
from src.data.targets import build_anchors
from src.evaluate import RATE_FLOOR, implied_dte, log_experiment, metrics
from src.models.baselines import GlobalAverage, TripSoFar


def constant_trip(seconds: float, trip_id: str = "TripT01", soc_per_km: float = 1.0) -> pd.DataFrame:
    """36 km/h (1 km per 100 s), SoC falling `soc_per_km` per km, soc_min 8 %."""
    n = int(seconds * 10)
    df = pd.DataFrame({
        "trip_id": trip_id, "series": "A", "segment": 0, "time_s": np.arange(n) * 0.1,
        "speed_kmh": 36.0, "battery_voltage_v": 400.0, "battery_current_a": -50.0,
        "soc_pct": 80 - np.arange(n) / 1000 * soc_per_km, "soc_min_pct": 8.0,
    })
    return add_derived_columns(df)


# ---- splits ---------------------------------------------------------------------------

def _fake_trips() -> pd.DataFrame:
    rows = []
    for series, n_days in [("A", 16), ("B", 19)]:
        for d in range(n_days):
            for k in range(1 + d % 3):  # 1-3 trips per day
                rows.append({"trip_id": f"Trip{series}{d:02d}{k}", "series": series, "day": f"{series}-{d:02d}"})
    return pd.DataFrame(rows)


def test_splits_keep_days_together_and_stratify() -> None:
    s = make_splits(_fake_trips(), test_frac=0.2, n_folds=5, seed=0)
    assert s.groupby("day")["fold"].nunique().max() == 1          # a day never straddles groups
    test_days = s[s.split == "test"].groupby("series")["day"].nunique()
    assert test_days.to_dict() == {"A": 3, "B": 4}                 # round(0.2*16), round(0.2*19)
    cv = s[s.split == "cv"]
    assert set(cv["fold"]) == set(range(5))
    assert (cv.groupby("fold")["series"].nunique() == 2).all()     # every fold has both seasons


def test_splits_are_deterministic() -> None:
    t = _fake_trips()
    pd.testing.assert_frame_equal(make_splits(t, seed=1), make_splits(t, seed=1))


# ---- targets --------------------------------------------------------------------------

def test_anchors_target_and_history() -> None:
    a = build_anchors(constant_trip(801.0), horizon_km=5.0, step_km=0.1, min_history_km=1.0)
    # ~8.01 km segment: anchors at 1.0 ... 3.0 km (each needs 5 km ahead)
    assert len(a) == 21
    assert a["km_in_segment"].min() == pytest.approx(1.0, abs=0.01)
    assert a["rate_next"].to_numpy() == pytest.approx(1.0, rel=0.01)
    assert a["so_far_rate"].to_numpy() == pytest.approx(1.0, rel=0.01)
    assert a["dte_true_km"].to_numpy() == pytest.approx(a["usable_soc_pct"].to_numpy(), rel=0.01)


def test_short_trip_gives_no_anchors_but_others_do() -> None:
    clean = pd.concat([constant_trip(500.0, "TripT01"), constant_trip(800.0, "TripT02")], ignore_index=True)
    a = build_anchors(clean, horizon_km=5.0)
    assert set(a["trip_id"]) == {"TripT02"}  # 5 km < 1 km history + 5 km horizon


# ---- baselines ------------------------------------------------------------------------

def test_global_average_is_distance_weighted() -> None:
    clean = pd.concat([constant_trip(1000.0, "TripT01", soc_per_km=1.0),   # 10 km at 1.0
                       constant_trip(3000.0, "TripT02", soc_per_km=0.5)],  # 30 km at 0.5
                      ignore_index=True)
    rate = GlobalAverage().fit(clean).rate_
    assert rate == pytest.approx((10 * 1.0 + 30 * 0.5) / 40, rel=0.01)


def test_trip_so_far_returns_so_far_rate() -> None:
    anchors = pd.DataFrame({"so_far_rate": [0.4, 0.9]})
    assert TripSoFar().fit(None).predict(anchors).tolist() == [0.4, 0.9]


# ---- metrics --------------------------------------------------------------------------

def test_metrics_known_values() -> None:
    m = metrics(rate_true=[1.0, 1.0], rate_pred=[0.5, 2.0], usable_soc_pct=[50.0, 50.0])
    assert m["mae_rate"] == pytest.approx(0.75)
    assert m["rmse_rate"] == pytest.approx(np.sqrt((0.25 + 1.0) / 2))
    assert m["mae_dte_km"] == pytest.approx((50 + 25) / 2)  # true DTE 50; predicted 100 and 25


def test_implied_dte_floors_nonpositive_rates() -> None:
    assert implied_dte([40.0], [-0.2])[0] == pytest.approx(40.0 / RATE_FLOOR)


def test_log_experiment_appends_rows(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    for i in range(2):
        log_experiment(f"run{i}", "2", "m", "cv", {"a": 1}, {"mae_rate": 0.1, "rmse_rate": 0.2, "mae_dte_km": 3.0}, path=path)
    log = pd.read_csv(path)
    assert len(log) == 2 and log["run_id"].tolist() == ["run0", "run1"]
