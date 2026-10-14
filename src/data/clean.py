"""Clean trips loaded by `src.data.load`.

Each step targets a specific problem found while auditing the raw data (see data/DATA_CARD.md):

1. `mask_interpolated`  - logger gaps that were filled with straight lines (TripB02 start,
                          TripB34 middle-to-end). Fabricated samples, not measurements.
2. `mask_soc_glitches`  - single-sample SoC jumps that are physically impossible
                          (TripB14's last row reads 0 % right after 34.6 %).
3. `mask_soc_frozen`    - SoC stuck at one value while current is flowing (TripB07: frozen
                          for ~5 min, then missing). The BMS signal is unreliable there.
4. `mask_temp_spikes`   - single-sample battery temperature spikes (TripA10: 60.9 °C, then 23 °C).
5. `mask_charging`      - standing still with large positive current = charging stop
                          (TripB04, TripB09). Not driving, and it raises SoC.
6. Rows flagged by 1-5, and rows with missing SoC, are dropped. Gaps left behind split a
   trip into `segment`s; segments shorter than `MIN_SEGMENT_S` are dropped.
7. `heater_standstill_check` - column-level, not row-level: if a trip reports more heater
                          power at standstill than the whole battery supplies (TripA06, A05,
                          A13), its `heater_power_kw` is set to NaN for that trip. Rows are kept.

Sign convention (verified in the data card): battery current is NEGATIVE when discharging.
"""

import numpy as np
import pandas as pd

from src.data.load import SAMPLE_PERIOD_S

INTERP_MIN_S = 5.0        # straight-line run must last at least this long to count as filled
# |second difference| at or below this = linear. Values are stored with 2 decimals, so a
# rounded straight line has |d2| <= 0.02. On this dataset, 0.021 flags only TripB02 and
# TripB34; the first false positives (short 6-7 s runs in A15/A30) appear at 0.05.
INTERP_TOL = 0.021
SOC_JUMP_PCT = 1.0        # max plausible SoC change in 0.1 s; real steps are <= 0.06 %
SOC_FROZEN_S = 120.0      # SoC unchanged this long ...
SOC_FROZEN_MIN_A = 10.0   # ... while mean |current| exceeds this = frozen sensor
TEMP_SPIKE_C = 5.0        # battery temp this far from its local median = sensor spike
CHARGE_MIN_A = 10.0       # positive current above this while stationary = charging
CHARGE_MIN_S = 60.0       # ... sustained at least this long
STATIONARY_KMH = 1.0
MIN_SEGMENT_S = 60.0      # drop leftover segments shorter than this
STANDSTILL_KMH = 0.5      # heater check: "standing still" ...
HEATER_CHECK_MIN_S = 30.0 # ... for at least this long in total, else the check is skipped


def runs(mask: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    """Return (start, stop) index pairs (stop exclusive) of True-runs with length >= min_len."""
    padded = np.r_[False, np.asarray(mask, bool), False]
    edges = np.diff(padded.astype(int))
    starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, stops) if e - s >= min_len]


def _runs_to_mask(n: int, spans: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(n, bool)
    for s, e in spans:
        mask[s:e] = True
    return mask


def mask_interpolated(trip: pd.DataFrame) -> np.ndarray:
    """Flag samples where speed AND battery current are both linear (up to rounding) for >= INTERP_MIN_S.

    Real 10 Hz sensor data is never linear in two independent signals at once while driving;
    a straight line in both means the gap was filled by interpolation.
    """
    n = len(trip)
    linear = np.zeros(n, bool)
    if n >= 3:
        d2v = np.abs(np.diff(trip["speed_kmh"].to_numpy(), 2))
        d2i = np.abs(np.diff(trip["battery_current_a"].to_numpy(), 2))
        linear[1:-1] = (d2v <= INTERP_TOL) & (d2i <= INTERP_TOL)
        # exclude standing still (speed constant at 0 is legitimately "linear")
        linear &= trip["speed_kmh"].to_numpy() > STATIONARY_KMH
    spans = runs(linear, int(INTERP_MIN_S / SAMPLE_PERIOD_S))
    # the run's two end points are part of the straight line too
    return _runs_to_mask(n, [(max(s - 1, 0), min(e + 1, n)) for s, e in spans])


def mask_soc_glitches(trip: pd.DataFrame) -> np.ndarray:
    """Flag samples whose SoC differs from the previous valid SoC by more than SOC_JUMP_PCT."""
    soc = trip["soc_pct"]
    prev = soc.ffill().shift(1)
    return ((soc - prev).abs() > SOC_JUMP_PCT).to_numpy()


def mask_soc_frozen(trip: pd.DataFrame) -> np.ndarray:
    """Flag runs where SoC is unchanged for >= SOC_FROZEN_S while mean |current| > SOC_FROZEN_MIN_A.

    Parking with the car on draws a few amps and SoC legitimately stays flat, hence the
    current condition.
    """
    soc = trip["soc_pct"].to_numpy()
    current = trip["battery_current_a"].to_numpy()
    same = np.r_[False, soc[1:] == soc[:-1]]
    spans = []
    for s, e in runs(same, int(SOC_FROZEN_S / SAMPLE_PERIOD_S)):
        s = s - 1  # include the first sample of the flat stretch
        if np.nanmean(np.abs(current[s:e])) > SOC_FROZEN_MIN_A:
            spans.append((s, e))
    return _runs_to_mask(len(trip), spans)


def mask_temp_spikes(trip: pd.DataFrame) -> np.ndarray:
    """Flag battery-temperature samples more than TEMP_SPIKE_C from the median of their 1 s neighbourhood.

    A battery pack cannot change temperature by 5 °C within a second (TripA10's first row reads
    60.9 °C, the next 23.0 °C). A centred rolling median keeps the neighbours of a spike unflagged.
    """
    flagged = np.zeros(len(trip), bool)
    for col in ("battery_temp_c", "battery_temp_max_c"):
        local = trip[col].rolling(11, center=True, min_periods=1).median()
        flagged |= ((trip[col] - local).abs() > TEMP_SPIKE_C).to_numpy()
    return flagged


def mask_charging(trip: pd.DataFrame) -> np.ndarray:
    """Flag sustained charging: stationary with current > CHARGE_MIN_A for >= CHARGE_MIN_S."""
    charging = (trip["speed_kmh"] < STATIONARY_KMH) & (trip["battery_current_a"] > CHARGE_MIN_A)
    return _runs_to_mask(len(trip), runs(charging.to_numpy(), int(CHARGE_MIN_S / SAMPLE_PERIOD_S)))


def heater_standstill_check(trip: pd.DataFrame) -> tuple[float, float]:
    """Median heater power and median battery draw (kW) over the trip's standstill samples.

    Standing still, the battery only feeds auxiliaries, so heater power <= battery draw must
    hold. Returns (nan, nan) if the trip has less than HEATER_CHECK_MIN_S of standstill.
    """
    still = trip[trip["speed_kmh"] < STANDSTILL_KMH]
    if len(still) * SAMPLE_PERIOD_S < HEATER_CHECK_MIN_S:
        return float("nan"), float("nan")
    battery_kw = -still["battery_voltage_v"] * still["battery_current_a"] / 1000.0
    return float(still["heater_power_kw"].median()), float(battery_kw.median())


def assign_segments(time_s: pd.Series) -> np.ndarray:
    """Number contiguous stretches of samples; a new segment starts wherever a time gap appears."""
    gap = time_s.diff().to_numpy() > 1.5 * SAMPLE_PERIOD_S
    return np.cumsum(gap)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add physically derived columns used throughout the project.

    - `power_draw_kw`: electrical power leaving the battery, kW (= -V*I/1000; positive = discharging)
    - `distance_km`: distance driven since the start of the trip, from integrated speed
    """
    df = df.copy()
    df["power_draw_kw"] = -df["battery_voltage_v"] * df["battery_current_a"] / 1000.0
    step_km = df["speed_kmh"].clip(lower=0) / 3600.0 * SAMPLE_PERIOD_S
    df["distance_km"] = step_km.groupby(df["trip_id"]).cumsum()
    return df


def clean_trip(trip: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply all cleaning steps to one trip. Returns the cleaned trip and a row-count report."""
    trip = trip.reset_index(drop=True)
    n = len(trip)
    steps = {
        "interpolated": mask_interpolated(trip),
        "soc_glitch": mask_soc_glitches(trip),
        "soc_frozen": mask_soc_frozen(trip),
        "temp_spike": mask_temp_spikes(trip),
        "charging": mask_charging(trip),
        "soc_missing": trip["soc_pct"].isna().to_numpy(),
    }
    report = {"trip_id": trip["trip_id"].iat[0], "rows_raw": n}
    drop = np.zeros(n, bool)
    for name, mask in steps.items():  # count each row once, under the first step that flags it
        report[f"drop_{name}"] = int((mask & ~drop).sum())
        drop |= mask
    kept = trip.loc[~drop].copy()

    kept["segment"] = assign_segments(kept["time_s"])
    seg_len_s = kept.groupby("segment")["time_s"].transform("size") * SAMPLE_PERIOD_S
    short = seg_len_s < MIN_SEGMENT_S
    report["drop_short_segment"] = int(short.sum())
    kept = kept.loc[~short]
    kept["segment"] = assign_segments(kept["time_s"])  # renumber 0..k-1

    report["rows_clean"] = len(kept)
    report["n_segments"] = int(kept["segment"].nunique())

    heater_kw, battery_kw = heater_standstill_check(kept)
    report["heater_still_kw"], report["battery_still_kw"] = heater_kw, battery_kw
    report["heater_kwh_reported"] = float(kept["heater_power_kw"].sum() * SAMPLE_PERIOD_S / 3600)
    report["heater_nulled"] = bool(heater_kw > battery_kw)
    if report["heater_nulled"]:
        kept["heater_power_kw"] = np.nan
    return kept, report


def clean_all(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean every trip in a long DataFrame from `load_all`. Returns (clean_df, per-trip report)."""
    parts, reports = [], []
    for _, trip in df.groupby("trip_id", sort=True):
        kept, rep = clean_trip(trip)
        parts.append(kept)
        reports.append(rep)
    clean = add_derived_columns(pd.concat(parts, ignore_index=True))
    return clean, pd.DataFrame(reports)
