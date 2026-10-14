"""Tabular features for Ridge and LightGBM, computed at each anchor from PAST data only.

"Last 1 km" windows always exist: every anchor has >= 1 km of history (src/data/targets.py).
Window sums use cumulative sums over each trip segment, so building all features is O(rows).

Feature sets (Phase 3 ablation):
- SET_1: the project brief's list.
- SET_2: SET_1 + trip history (so-far rate, recent consumption) + physics (heater cost per km)
         + battery temperature + A/C power.
"""

import numpy as np
import pandas as pd

from src.data.load import SAMPLE_PERIOD_S

WINDOW_KM = 1.0
MIN_SPEED_KMH = 5.0  # floor for heater kWh/km so a long standstill doesn't divide by ~0

SET_1 = ["speed_mean_1km", "speed_std_1km", "accel_pos_mean_1km", "elev_delta_1km", "ambient_temp_c",
         "elapsed_min", "km_in_segment", "soc_pct", "heater_kw_1km"]
SET_2 = SET_1 + ["so_far_rate", "recent_kwh_per_km_1km", "battery_temp_c", "heater_kwh_per_km", "aircon_kw_1km"]
FEATURE_SETS = {"set1": SET_1, "set2": SET_2}


def _window_mean(cs: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Mean of the samples in [lo, hi) given a cumulative sum `cs` with a leading 0."""
    return (cs[hi] - cs[lo]) / (hi - lo)


def tabular_features(clean: pd.DataFrame, anchors: pd.DataFrame, window_km: float = WINDOW_KM) -> pd.DataFrame:
    """Return a DataFrame aligned with `anchors` holding every feature in SET_2 (SET_1 is a subset).

    `anchors["row"]` is the positional index of the anchor sample in `clean` (see build_anchors).
    Windows cover the samples in the last `window_km` before the anchor, excluding the anchor itself.
    """
    clean = clean.reset_index(drop=True)
    out = pd.DataFrame(index=anchors.index)
    seg_key = clean["trip_id"] + "#" + clean["segment"].astype(str)
    a_key = anchors["trip_id"] + "#" + anchors["segment"].astype(str)

    cols = {k: np.full(len(anchors), np.nan) for k in
            ["speed_mean_1km", "speed_std_1km", "accel_pos_mean_1km", "elev_delta_1km", "elapsed_min",
             "heater_kw_1km", "recent_kwh_per_km_1km", "aircon_kw_1km"]}
    for key, s in clean.groupby(seg_key, sort=False):
        mask = (a_key == key).to_numpy()
        if not mask.any():
            continue
        first = s.index[0]
        i0 = anchors.loc[mask, "row"].to_numpy() - first          # anchor positions within the segment
        x = s["distance_km"].to_numpy()
        lo = np.searchsorted(x, x[i0] - window_km)                # first sample inside the window

        def cs(v: np.ndarray) -> np.ndarray:
            return np.r_[0.0, np.cumsum(v)]

        v = s["speed_kmh"].to_numpy()
        mean_v = _window_mean(cs(v), lo, i0)
        cols["speed_mean_1km"][mask] = mean_v
        cols["speed_std_1km"][mask] = np.sqrt(np.maximum(_window_mean(cs(v ** 2), lo, i0) - mean_v ** 2, 0))
        cols["accel_pos_mean_1km"][mask] = _window_mean(cs(s["accel_long_ms2"].clip(lower=0).to_numpy()), lo, i0)
        elev = s["elevation_m"].to_numpy()
        cols["elev_delta_1km"][mask] = elev[i0] - elev[lo]
        t = s["time_s"].to_numpy()
        cols["elapsed_min"][mask] = (t[i0] - t[0]) / 60
        cols["heater_kw_1km"][mask] = _window_mean(cs(s["heater_power_kw"].to_numpy()), lo, i0)
        cols["aircon_kw_1km"][mask] = _window_mean(cs(s["aircon_power_kw"].to_numpy()), lo, i0)
        kwh = cs(s["power_draw_kw"].to_numpy() * SAMPLE_PERIOD_S / 3600)
        cols["recent_kwh_per_km_1km"][mask] = (kwh[i0] - kwh[lo]) / (x[i0] - x[lo])

    for k, v in cols.items():
        out[k] = v
    row = anchors["row"].to_numpy()
    out["ambient_temp_c"] = clean["ambient_temp_c"].to_numpy()[row]
    out["battery_temp_c"] = clean["battery_temp_c"].to_numpy()[row]
    out["soc_pct"] = anchors["soc_pct"].to_numpy()
    out["km_in_segment"] = anchors["km_in_segment"].to_numpy()
    out["so_far_rate"] = anchors["so_far_rate"].to_numpy()
    out["heater_kwh_per_km"] = out["heater_kw_1km"] / np.maximum(out["speed_mean_1km"], MIN_SPEED_KMH)
    return out[SET_2]
