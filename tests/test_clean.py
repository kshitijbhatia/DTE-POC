"""Tests for src.data.clean using small synthetic trips."""

import numpy as np
import pandas as pd
import pytest

from src.data.clean import (
    add_derived_columns,
    clean_trip,
    mask_charging,
    mask_interpolated,
    mask_soc_frozen,
    mask_soc_glitches,
    mask_temp_spikes,
    runs,
)


def make_trip(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """A plausible noisy driving trip: ~40 km/h, ~-30 A, SoC slowly falling, 10 Hz."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "trip_id": "TripT01",
        "time_s": np.round(np.arange(n) * 0.1, 1),
        "speed_kmh": np.round(40 + rng.normal(0, 1, n), 2),
        "battery_current_a": np.round(-30 + rng.normal(0, 3, n), 2),
        "battery_voltage_v": 380.0,
        "soc_pct": np.round(80 - np.arange(n) * 0.001, 1),
        "battery_temp_c": 20.0,
        "battery_temp_max_c": 21.0,
        "heater_power_kw": 0.5,
    })


def test_runs_finds_true_stretches() -> None:
    mask = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1], bool)
    assert runs(mask) == [(1, 3), (4, 7), (8, 9)]
    assert runs(mask, min_len=3) == [(4, 7)]


def test_mask_interpolated_flags_rounded_linear_fill_only() -> None:
    trip = make_trip()
    assert not mask_interpolated(trip).any()  # noisy real-looking data: nothing flagged
    # fill rows 1000-1999 with a straight line rounded to 2 decimals, like TripB02
    trip.loc[1000:1999, "speed_kmh"] = np.round(np.linspace(20, 60, 1000), 2)
    trip.loc[1000:1999, "battery_current_a"] = np.round(np.linspace(-10, -90, 1000), 2)
    flagged = np.where(mask_interpolated(trip))[0]
    assert flagged.min() >= 999 and flagged.max() <= 2000
    assert len(flagged) >= 990


def test_mask_interpolated_ignores_standing_still() -> None:
    trip = make_trip()
    trip["speed_kmh"] = 0.0
    trip["battery_current_a"] = -2.0
    assert not mask_interpolated(trip).any()


def test_mask_soc_glitches_flags_impossible_jump() -> None:
    trip = make_trip()
    trip.loc[2999, "soc_pct"] = 0.0  # like TripB14's final row
    assert np.where(mask_soc_glitches(trip))[0].tolist() == [2999]


def test_mask_soc_frozen_requires_current_flow() -> None:
    trip = make_trip()
    trip.loc[500:2500, "soc_pct"] = 70.0  # 200 s flat while drawing ~30 A -> frozen
    assert mask_soc_frozen(trip)[500:2501].all()

    parked = make_trip()
    parked.loc[500:2500, "soc_pct"] = 70.0
    parked.loc[500:2500, "battery_current_a"] = -3.0  # flat SoC while parked is legitimate
    assert not mask_soc_frozen(parked).any()


def test_mask_temp_spikes_flags_only_the_spike() -> None:
    trip = make_trip()
    trip["battery_temp_c"] = 22.0
    trip["battery_temp_max_c"] = 23.0
    trip.loc[0, "battery_temp_max_c"] = 60.9   # like TripA10's first row
    trip.loc[2000:, "battery_temp_c"] = 23.0   # a normal 1 °C step must not be flagged
    assert np.where(mask_temp_spikes(trip))[0].tolist() == [0]


def test_mask_charging_needs_stationary_sustained_positive_current() -> None:
    trip = make_trip()
    trip.loc[1000:1999, ["speed_kmh", "battery_current_a"]] = [0.0, 90.0]  # 100 s charging
    trip.loc[2500:2549, "battery_current_a"] = 50.0  # 5 s of regen while moving: not charging
    flagged = mask_charging(trip)
    assert flagged[1000:2000].all() and flagged.sum() == 1000


def test_clean_trip_drops_rows_splits_segments_and_reports() -> None:
    trip = make_trip(n=4000)
    trip.loc[1000:1999, ["speed_kmh", "battery_current_a"]] = [0.0, 90.0]  # charging stop
    trip.loc[3000:3449, "soc_pct"] = np.nan  # missing SoC ...
    trip.loc[3500:3999, "soc_pct"] = np.nan  # ... leaving an isolated 5 s piece (3450-3499)
    clean, report = clean_trip(trip)
    assert report["drop_charging"] == 1000
    assert report["drop_soc_missing"] == 950
    assert report["drop_short_segment"] == 50
    assert report["n_segments"] == 2  # before / after the charging stop
    assert report["rows_raw"] - sum(v for k, v in report.items() if k.startswith("drop_")) == len(clean)
    assert clean["soc_pct"].notna().all()


def test_clean_trip_nulls_implausible_heater_but_keeps_rows() -> None:
    trip = make_trip()
    trip["heater_power_kw"] = 1.0
    trip.loc[:999, ["speed_kmh", "battery_current_a"]] = [0.0, -3.0]  # 100 s parked: ~1.1 kW draw
    clean, report = clean_trip(trip)
    assert not report["heater_nulled"] and clean["heater_power_kw"].notna().all()

    trip["heater_power_kw"] = 5.3  # more than the battery supplies while parked: impossible
    clean, report = clean_trip(trip)
    assert report["heater_nulled"] and clean["heater_power_kw"].isna().all()
    assert len(clean) == len(trip)


def test_add_derived_columns_distance_and_power_sign() -> None:
    df = pd.DataFrame({
        "trip_id": "TripT01",
        "speed_kmh": [36.0] * 1000,          # 10 m/s for 100 s = 1 km
        "battery_voltage_v": [400.0] * 1000,
        "battery_current_a": [-50.0] * 1000,  # discharging
    })
    out = add_derived_columns(df)
    assert out["distance_km"].iat[-1] == pytest.approx(1.0)
    assert out["power_draw_kw"].iat[0] == pytest.approx(20.0)  # positive = energy leaving battery
