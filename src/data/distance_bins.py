"""Aggregate the 10 Hz cleaned data into fixed-distance bins (e.g. one row per driven km).

Consumption is only meaningful over a stretch of road: per-sample SoC changes are tiny and
quantised, and per-sample power swings between +140 kW and -55 kW. Binning by distance gives
one consumption value per stretch, plus the driving conditions over that stretch.
"""

import numpy as np
import pandas as pd

from src.data.load import SAMPLE_PERIOD_S


def distance_bins(df: pd.DataFrame, bin_km: float = 1.0, min_cover: float = 0.95) -> pd.DataFrame:
    """Split each trip segment into consecutive `bin_km` stretches and summarise each one.

    Bins never cross a segment boundary (e.g. TripB04's charging stop). The last, partial bin
    of a segment is kept only if it covers at least `min_cover * bin_km`.

    Returns one row per bin with: trip_id, series, segment, bin, distance_km, duration_s,
    soc_start_pct, soc_drop_pct, energy_kwh, soc_per_km, kwh_per_100km, speed_mean_kmh,
    speed_std_kmh, ambient_temp_c, battery_temp_c, heater_kw, elev_delta_m, grade_pct.
    """
    d = df[["trip_id", "series", "segment", "distance_km", "soc_pct", "power_draw_kw", "speed_kmh",
            "ambient_temp_c", "battery_temp_c", "heater_power_kw", "elevation_m"]].copy()
    seg_start = d.groupby(["trip_id", "segment"])["distance_km"].transform("first")
    d["bin"] = np.floor((d["distance_km"] - seg_start) / bin_km).astype(int)
    d["energy_kwh"] = d["power_draw_kw"] * SAMPLE_PERIOD_S / 3600.0

    g = d.groupby(["trip_id", "segment", "bin"], sort=True)
    out = g.agg(
        series=("series", "first"),
        dist_first=("distance_km", "first"),
        dist_last=("distance_km", "last"),
        n=("distance_km", "size"),
        soc_start_pct=("soc_pct", "first"),
        soc_end_pct=("soc_pct", "last"),
        energy_kwh=("energy_kwh", "sum"),
        speed_mean_kmh=("speed_kmh", "mean"),
        speed_std_kmh=("speed_kmh", "std"),
        ambient_temp_c=("ambient_temp_c", "mean"),
        battery_temp_c=("battery_temp_c", "mean"),
        heater_kw=("heater_power_kw", "mean"),
        elev_first=("elevation_m", "first"),
        elev_last=("elevation_m", "last"),
    ).reset_index()

    out["distance_km"] = out["dist_last"] - out["dist_first"]
    out["duration_s"] = out["n"] * SAMPLE_PERIOD_S
    out["soc_drop_pct"] = out["soc_start_pct"] - out["soc_end_pct"]
    out["elev_delta_m"] = out["elev_last"] - out["elev_first"]
    out = out[out["distance_km"] >= min_cover * bin_km].copy()
    out["soc_per_km"] = out["soc_drop_pct"] / out["distance_km"]
    out["kwh_per_100km"] = out["energy_kwh"] / out["distance_km"] * 100
    out["grade_pct"] = out["elev_delta_m"] / (out["distance_km"] * 1000) * 100
    cols = ["trip_id", "series", "segment", "bin", "distance_km", "duration_s", "soc_start_pct",
            "soc_drop_pct", "energy_kwh", "soc_per_km", "kwh_per_100km", "speed_mean_kmh",
            "speed_std_kmh", "ambient_temp_c", "battery_temp_c", "heater_kw", "elev_delta_m", "grade_pct"]
    return out[cols].reset_index(drop=True)
