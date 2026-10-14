"""Prediction points ("anchors") and the target: consumption rate over the next N km.

At each anchor (every `step_km` along a trip segment) we know everything up to that point and
predict

    rate = (SoC now - SoC after N more km) / N            [%SoC per km]

From a predicted rate, distance-to-empty follows as

    DTE = usable SoC now / rate,   usable SoC = SoC - soc_min (the BMS floor, 8 %)   [km]

No trip reaches empty, so the "actual" DTE used for evaluation is usable SoC / actual rate over
the next N km: the range the car would have if the next N km's consumption continued.
"""

import numpy as np
import pandas as pd

HORIZON_KM = 5.0      # N, chosen in Phase 2 (see README)
STEP_KM = 0.1         # one anchor every 100 m
MIN_HISTORY_KM = 1.0  # anchors need this much driving behind them in the segment


def build_anchors(clean: pd.DataFrame, horizon_km: float = HORIZON_KM, step_km: float = STEP_KM,
                  min_history_km: float = MIN_HISTORY_KM) -> pd.DataFrame:
    """Return one row per anchor with the target and what is known at that moment.

    Columns: trip_id, series, segment, row (positional index into `clean`), time_s,
    km_in_segment, soc_pct, usable_soc_pct, so_far_rate (segment's average %SoC/km up to the
    anchor), rate_next (target, %SoC/km over the next `horizon_km`), dte_true_km.
    Anchors never look across a segment boundary (e.g. TripB04's charging stop).
    """
    clean = clean.reset_index(drop=True)
    parts = []
    for (tid, seg), s in clean.groupby(["trip_id", "segment"], sort=True):
        x = s["distance_km"].to_numpy() - s["distance_km"].iat[0]
        soc = s["soc_pct"].to_numpy()
        anchors_km = np.arange(min_history_km, x[-1] - horizon_km + 1e-9, step_km)
        if len(anchors_km) == 0:
            continue
        i0 = np.searchsorted(x, anchors_km)                 # first sample at/after the anchor
        i1 = np.searchsorted(x, x[i0] + horizon_km)         # first sample N km later
        i1 = np.minimum(i1, len(x) - 1)
        rate_next = (soc[i0] - soc[i1]) / (x[i1] - x[i0])
        so_far = (soc[0] - soc[i0]) / x[i0]
        usable = soc[i0] - s["soc_min_pct"].to_numpy()[i0]
        parts.append(pd.DataFrame({
            "trip_id": tid, "series": s["series"].iat[0], "segment": seg,
            "row": s.index.to_numpy()[i0], "time_s": s["time_s"].to_numpy()[i0],
            "km_in_segment": x[i0], "soc_pct": soc[i0], "usable_soc_pct": usable,
            "so_far_rate": so_far, "rate_next": rate_next, "dte_true_km": usable / rate_next,
        }))
    return pd.concat(parts, ignore_index=True)


def horizon_conditions(clean: pd.DataFrame, anchors: pd.DataFrame, horizon_km: float = HORIZON_KM) -> pd.DataFrame:
    """Driving conditions over each anchor's NEXT `horizon_km` -- for ERROR ANALYSIS ONLY.

    These describe the future stretch being predicted, so they must never be used as model
    inputs (that would be look-ahead). Returns, aligned with `anchors`: future_speed_kmh (mean),
    future_grade_pct (elevation change / distance), future_abs_grade_pct (mean |grade| over
    0.5 km pieces, a hilliness measure), future_heater_kw (mean).
    """
    clean = clean.reset_index(drop=True)
    out = pd.DataFrame(index=anchors.index, columns=["future_speed_kmh", "future_grade_pct",
                                                     "future_abs_grade_pct", "future_heater_kw"], dtype=float)
    seg_key = clean["trip_id"] + "#" + clean["segment"].astype(str)
    a_key = (anchors["trip_id"] + "#" + anchors["segment"].astype(str)).to_numpy()
    for key, s in clean.groupby(seg_key, sort=False):
        idx = np.where(a_key == key)[0]
        if len(idx) == 0:
            continue
        x = s["distance_km"].to_numpy()
        elev = s["elevation_m"].to_numpy()
        cs_v = np.r_[0.0, np.cumsum(s["speed_kmh"].to_numpy())]
        cs_h = np.r_[0.0, np.cumsum(s["heater_power_kw"].to_numpy())]
        i0 = anchors["row"].to_numpy()[idx] - s.index[0]
        i1 = np.minimum(np.searchsorted(x, x[i0] + horizon_km), len(x) - 1)
        out.iloc[idx, 0] = (cs_v[i1] - cs_v[i0]) / (i1 - i0)
        out.iloc[idx, 1] = (elev[i1] - elev[i0]) / ((x[i1] - x[i0]) * 1000) * 100
        # hilliness: mean |grade| over consecutive 0.5 km pieces of the horizon
        pieces = np.arange(0, horizon_km + 1e-9, 0.5)
        j = np.minimum(np.searchsorted(x, x[i0][:, None] + pieces[None, :]), len(x) - 1)
        dz, dx = np.diff(elev[j], axis=1), np.diff(x[j], axis=1) * 1000
        out.iloc[idx, 2] = np.mean(np.abs(dz) / np.maximum(dx, 1.0), axis=1) * 100
        out.iloc[idx, 3] = (cs_h[i1] - cs_h[i0]) / (i1 - i0)
    return out.astype(float)
