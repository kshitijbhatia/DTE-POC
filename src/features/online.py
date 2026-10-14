"""Compute model features for ONE prediction "now", from a live telemetry window.

Used by the server (and later the app). It deliberately reuses `tabular_features` -- the exact
code that produced the training features -- on a one-trip, one-anchor frame, then overrides the
three trip-level values that a short window cannot know (time and distance since trip start,
trip-average rate so far) with values the client tracks. tests/test_server.py checks that this
reproduces the offline features exactly.
"""

import numpy as np
import pandas as pd

from src.data.load import SAMPLE_PERIOD_S
from src.features.build import SET_2, WINDOW_KM, tabular_features

SAMPLE_FIELDS = ["time_s", "speed_kmh", "accel_long_ms2", "elevation_m", "battery_voltage_v", "battery_current_a",
                 "soc_pct", "ambient_temp_c", "battery_temp_c", "heater_power_kw", "aircon_power_kw"]


def window_distance_km(samples: pd.DataFrame) -> float:
    """Distance covered by the window BEFORE the last ("now") sample, as integrated speed."""
    step_km = samples["speed_kmh"].clip(lower=0) / 3600.0 * SAMPLE_PERIOD_S
    return float(step_km.iloc[1:].sum())


def features_now(samples: pd.DataFrame, elapsed_s: float, km_since_start: float,
                 soc_at_start_pct: float) -> pd.Series:
    """Return the SET_2 feature vector for the last sample of `samples` (10 Hz, oldest first).

    The window must cover at least WINDOW_KM before the last sample (ValueError otherwise).
    """
    if window_distance_km(samples) < WINDOW_KM:
        raise ValueError(f"window covers {window_distance_km(samples):.3f} km; need >= {WINDOW_KM} km before 'now'")
    clean = samples[SAMPLE_FIELDS].astype(float).reset_index(drop=True).copy()
    clean.insert(0, "trip_id", "live")
    clean["segment"] = 0
    clean["power_draw_kw"] = -clean["battery_voltage_v"] * clean["battery_current_a"] / 1000.0
    clean["distance_km"] = (clean["speed_kmh"].clip(lower=0) / 3600.0 * SAMPLE_PERIOD_S).cumsum()
    now = len(clean) - 1
    so_far = (soc_at_start_pct - clean["soc_pct"].iat[now]) / km_since_start
    anchor = pd.DataFrame({"trip_id": ["live"], "segment": [0], "row": [now], "soc_pct": [clean["soc_pct"].iat[now]],
                           "km_in_segment": [km_since_start], "so_far_rate": [so_far]})
    f = tabular_features(clean, anchor).iloc[0].copy()
    f["elapsed_min"] = elapsed_s / 60.0  # the window does not start at the trip start
    return f[SET_2]


def request_from_clean(clean: pd.DataFrame, anchor: pd.Series, margin_km: float = 0.2) -> dict:
    """Build a /predict request body for an offline anchor, as a live client would send it.

    Window = the samples covering the last 1 km (+ `margin_km`) up to and including the anchor sample.
    Trip context uses the anchor's segment start, matching the offline feature definitions.
    """
    seg = clean[(clean.trip_id == anchor.trip_id) & (clean.segment == anchor.segment)]
    x = seg.distance_km.to_numpy()
    pos = int(anchor.row - seg.index[0])
    start = int(np.searchsorted(x, x[pos] - WINDOW_KM - margin_km))
    window = seg.iloc[max(start - 1, 0): pos + 1]
    return {"samples": window[SAMPLE_FIELDS].to_dict(orient="records"),
            "trip": {"elapsed_s": float(seg.time_s.iat[pos] - seg.time_s.iat[0]),
                     "km_since_start": float(anchor.km_in_segment),
                     "soc_at_start_pct": float(seg.soc_pct.iat[0])}}


def implied_dte_km(soc_pct: float, rate: float, soc_min_pct: float = 8.0, rate_floor: float = 0.05) -> float:
    """Distance-to-empty (km) from current SoC and a consumption rate, same rule as src.evaluate."""
    return float((soc_pct - soc_min_pct) / max(rate, rate_floor))


def is_complete(f: pd.Series) -> bool:
    """True if no feature is missing (NaN)."""
    return bool(np.isfinite(f.to_numpy(dtype=float)).all())
