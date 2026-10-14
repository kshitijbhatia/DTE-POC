"""Tests for tabular features and GRU windows on a synthetic constant-speed trip."""

import numpy as np
import pandas as pd
import pytest

from src.data.clean import add_derived_columns
from src.data.targets import build_anchors
from src.features.build import SET_1, SET_2, tabular_features
from src.features.sequences import N_STEPS, SEQ_CHANNELS, gru_inputs


def make_trip(seconds: float = 1000.0, heater_kw: float = 2.0) -> pd.DataFrame:
    """36 km/h, 20 kW draw, elevation rising 0.01 m per sample (= +10 m per km), 10 Hz."""
    n = int(seconds * 10)
    df = pd.DataFrame({
        "trip_id": "TripT01", "series": "B", "segment": 0, "time_s": np.arange(n) * 0.1,
        "speed_kmh": 36.0, "accel_long_ms2": 0.0, "battery_voltage_v": 400.0, "battery_current_a": -50.0,
        "soc_pct": 80 - np.arange(n) / 1000, "soc_min_pct": 8.0, "elevation_m": 500 + np.arange(n) * 0.01,
        "ambient_temp_c": 3.0, "battery_temp_c": 10.0, "heater_power_kw": heater_kw, "aircon_power_kw": 0.0,
    })
    return add_derived_columns(df)


def test_tabular_features_values() -> None:
    clean = make_trip()
    anchors = build_anchors(clean)
    X = tabular_features(clean, anchors)
    assert list(X.columns) == SET_2 and set(SET_1) <= set(SET_2)
    assert X["speed_mean_1km"].to_numpy() == pytest.approx(36.0)
    assert X["speed_std_1km"].to_numpy() == pytest.approx(0.0, abs=1e-6)
    assert X["elev_delta_1km"].to_numpy() == pytest.approx(10.0, abs=0.02)
    assert X["recent_kwh_per_km_1km"].to_numpy() == pytest.approx(20 * 100 / 3600, rel=0.01)  # 20 kW for 100 s per km
    assert X["heater_kwh_per_km"].to_numpy() == pytest.approx(2.0 / 36.0)
    first = X.iloc[0]
    assert first["elapsed_min"] == pytest.approx(100 / 60, abs=0.01)  # 1 km at 36 km/h = 100 s


def test_tabular_features_use_only_the_past() -> None:
    clean = make_trip()
    anchors = build_anchors(clean)
    before = tabular_features(clean, anchors)
    future = clean.copy()
    last_anchor_row = anchors["row"].max()
    future.loc[last_anchor_row:, ["speed_kmh", "heater_power_kw", "power_draw_kw"]] = [150.0, 7.0, 99.0]
    after = tabular_features(future, anchors)
    pd.testing.assert_frame_equal(before, after)  # changing data at/after the anchor changes nothing


def test_gru_window_shape_padding_and_no_lookahead() -> None:
    clean = make_trip()
    anchors = build_anchors(clean)
    seq, ctx = gru_inputs(clean, anchors)
    assert seq.shape == (len(anchors), N_STEPS, len(SEQ_CHANNELS)) and ctx.shape == (len(anchors), 4)
    speed = SEQ_CHANNELS.index("speed_kmh")
    elev = SEQ_CHANNELS.index("elev_rel_m")
    assert seq[:, :, speed] == pytest.approx(36.0)
    # first anchor sits at 1 km = 100 s of history < 300 s window: oldest blocks are padded with
    # the segment's first sample, whose elevation is 10 m below the anchor
    assert seq[0, 0, elev] == pytest.approx(-10.0, abs=0.01)
    # newest block ends just before the anchor: its mean elevation is slightly below the anchor
    assert -0.3 < seq[0, -1, elev] < 0


def test_gru_window_keeps_missing_heater_as_nan() -> None:
    clean = make_trip(heater_kw=np.nan)
    seq, _ = gru_inputs(clean, build_anchors(clean))
    assert np.isnan(seq[:, :, SEQ_CHANNELS.index("heater_power_kw")]).all()
