"""Tests for src.data.distance_bins on a synthetic constant-speed, constant-power trip."""

import numpy as np
import pandas as pd
import pytest

from src.data.clean import add_derived_columns
from src.data.distance_bins import distance_bins


def make_constant_trip(seconds: float = 350.0) -> pd.DataFrame:
    """36 km/h (1 km per 100 s), 400 V x -50 A = 20 kW draw, SoC falling 1 %-point per 100 s."""
    n = int(seconds * 10)
    df = pd.DataFrame({
        "trip_id": "TripT01", "series": "A", "segment": 0,
        "speed_kmh": 36.0, "battery_voltage_v": 400.0, "battery_current_a": -50.0,
        "soc_pct": 80 - np.arange(n) / 1000, "ambient_temp_c": 10.0, "battery_temp_c": 20.0,
        "heater_power_kw": 1.0, "elevation_m": 500 + np.arange(n) * 0.01,  # +10 m per km
    })
    return add_derived_columns(df)


def test_bins_have_expected_count_and_consumption() -> None:
    bins = distance_bins(make_constant_trip(350.0), bin_km=1.0)
    # 3.5 km driven -> 3 full bins; the 0.5 km remainder is below min_cover and dropped
    assert len(bins) == 3
    assert bins["kwh_per_100km"].to_numpy() == pytest.approx(20 * 100 / 3600 * 100, rel=0.01)
    assert bins["soc_per_km"].to_numpy() == pytest.approx(1.0, rel=0.01)
    assert bins["grade_pct"].to_numpy() == pytest.approx(1.0, rel=0.01)


def test_bins_do_not_cross_segments() -> None:
    a = make_constant_trip(250.0)
    b = make_constant_trip(250.0).assign(segment=1)
    b["distance_km"] += a["distance_km"].iat[-1] + 0.5  # gap between segments
    bins = distance_bins(pd.concat([a, b], ignore_index=True), bin_km=1.0)
    assert bins.groupby("segment").size().to_dict() == {0: 2, 1: 2}
