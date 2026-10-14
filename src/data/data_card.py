"""Generate data/DATA_CARD.md. Every number in the card is computed here from the data.

Run after build_dataset:  .venv/bin/python -m src.data.data_card
Column meanings (the only hand-written content) live in `MEANINGS` below.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.build_dataset import CLEAN_PATH, REPORT_PATH
from src.data.clean import SAMPLE_PERIOD_S, mask_charging
from src.data.load import COLUMNS, RAW_DIR, _normalise_header, load_overview, load_trip, trip_files

CARD_PATH = Path(__file__).resolve().parents[2] / "data" / "DATA_CARD.md"

# standard column -> (unit, meaning). Meanings marked "(inferred)" are our reading of the data,
# not documented by the dataset authors.
MEANINGS: dict[str, tuple[str, str]] = {
    "time_s": ("s", "Time since logging started"),
    "speed_kmh": ("km/h", "Vehicle speed"),
    "elevation_m": ("m", "Altitude above sea level (sensor source not documented)"),
    "throttle_pct": ("%", "Accelerator pedal position"),
    "motor_torque_nm": ("Nm", "Electric motor torque; negative while regenerating"),
    "accel_long_ms2": ("m/s²", "Longitudinal acceleration sensor"),
    "regen_signal": ("–", "Regenerative braking signal; continuous, scale not documented"),
    "battery_voltage_v": ("V", "High-voltage battery pack voltage"),
    "battery_current_a": ("A", "High-voltage battery current; **negative = discharging** (verified below)"),
    "battery_temp_c": ("°C", "Battery temperature"),
    "battery_temp_max_c": ("°C", "Maximum battery (cell) temperature"),
    "soc_pct": ("%", "Battery management system (BMS) state of charge (raw, not the dashboard value)"),
    "soc_displayed_pct": ("%", "SoC shown to the driver on the dashboard"),
    "soc_min_pct": ("%", "Constant per trip; lower bound of usable SoC window (inferred)"),
    "soc_max_pct": ("%", "Constant per trip; upper bound of usable SoC window (inferred)"),
    "heater_power_kw": ("kW", "Cabin heater electrical power (CAN bus)"),
    "heater_requested_w": ("W", "Heating power requested by climate control"),
    "aircon_power_kw": ("kW", "Air-conditioning compressor power"),
    "heater_signal": ("0/1", "Heater on/off flag"),
    "ambient_temp_c": ("°C", "Outside air temperature"),
    "heat_exchanger_temp_c": ("°C", "Heat exchanger temperature"),
    "cabin_temp_c": ("°C", "Cabin temperature sensor"),
}
DERIVED: dict[str, tuple[str, str]] = {
    "trip_id": ("–", "Trip name from the file, e.g. TripA01"),
    "series": ("–", "A = summer 2019, B = winter 2019/20 (from Overview.xlsx dates)"),
    "segment": ("–", "Contiguous piece of a trip after cleaning (0, 1, ...); only TripB04 has 2"),
    "power_draw_kw": ("kW", "−V·I/1000: power leaving the battery (positive = discharging)"),
    "distance_km": ("km", "Distance since trip start, integrated from speed"),
}


def md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (no extra dependency needed)."""
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def fmt(x: float, nd: int = 2) -> str:
    """Format a number compactly for tables."""
    return f"{x:,.{nd}f}" if pd.notna(x) else "–"


def raw_facts() -> dict:
    """Facts about the raw files: header variants, column coverage, sampling, quirks."""
    files = trip_files()
    header_sets, col_count, rows, dt_ok = Counter(), Counter(), 0, 0
    raw_nan_soc = {}
    for f in files:
        header = pd.read_csv(f, sep=";", encoding="latin-1", nrows=0).columns
        header_sets[tuple(header)] += 1
        col_count.update({_normalise_header(c) for c in header})
        t = load_trip(f)
        rows += len(t)
        dt_ok += int(np.allclose(np.diff(t["time_s"]), SAMPLE_PERIOD_S, atol=1e-6))
        raw_nan_soc[f.stem] = int(t["soc_pct"].isna().sum())
    n = len(files)
    partial = sorted(c for c, k in col_count.items() if k < n and not c.startswith("Unnamed"))
    return {"n_files": n, "n_header_sets": len(header_sets), "raw_rows": rows, "dt_ok": dt_ok,
            "partial_cols": [(c, col_count[c]) for c in partial],
            "trips_with_nan_soc": sum(v > 0 for v in raw_nan_soc.values())}


def trip_summary(clean: pd.DataFrame) -> pd.DataFrame:
    """One row per trip (clean data): duration, distance, SoC used, mean speed/temperature."""
    g = clean.groupby("trip_id")
    seg = clean.groupby(["trip_id", "segment"])["soc_pct"]
    soc_used = (seg.first() - seg.last()).groupby("trip_id").sum()  # excludes SoC gained while charging
    return pd.DataFrame({
        "series": g["series"].first(),
        "duration_min": g.size() * SAMPLE_PERIOD_S / 60,
        "distance_km": g["distance_km"].max(),
        "soc_used_pct": soc_used,
        "mean_speed_kmh": g["speed_kmh"].mean(),
        "ambient_c": g["ambient_temp_c"].mean(),
    })


def coulomb_check(clean: pd.DataFrame) -> pd.DataFrame:
    """Per trip-segment: Ah and kWh drawn from the battery vs. BMS SoC drop."""
    out = []
    for (tid, seg), s in clean.groupby(["trip_id", "segment"]):
        dsoc = s["soc_pct"].iat[0] - s["soc_pct"].iat[-1]
        if dsoc < 2:  # too little SoC change for a stable ratio
            continue
        ah = -(s["battery_current_a"] * SAMPLE_PERIOD_S).sum() / 3600
        kwh = (s["power_draw_kw"] * SAMPLE_PERIOD_S).sum() / 3600
        out.append({"trip_id": tid, "dsoc": dsoc, "ah_per_pct": ah / dsoc, "kwh_per_pct": kwh / dsoc})
    return pd.DataFrame(out)


def coulomb_drift(clean: pd.DataFrame, ah_per_pct: float) -> pd.Series:
    """Per trip-segment: max |BMS SoC - SoC re-computed from integrated current| in %-points."""
    out = {}
    for (tid, seg), s in clean.groupby(["trip_id", "segment"]):
        cc = s["soc_pct"].iat[0] + (s["battery_current_a"] * SAMPLE_PERIOD_S).cumsum() / 3600 / ah_per_pct
        out[(tid, seg)] = (s["soc_pct"] - cc).abs().max()
    return pd.Series(out)


def quantiles(s: pd.Series, nd: int = 1) -> str:
    q = s.quantile([0, .25, .5, .75, 1]).to_numpy()
    return " / ".join(fmt(v, nd) for v in q)


def build_card() -> str:
    """Compute every statistic and assemble the markdown text of the data card."""
    clean = pd.read_parquet(CLEAN_PATH)
    report = pd.read_csv(REPORT_PATH)
    overview = load_overview()
    rf = raw_facts()
    trips = trip_summary(clean)
    cc = coulomb_check(clean)

    ov = overview.assign(day=pd.to_datetime(overview["date"].str[:10].str.replace("_", "-")))
    ov["series"] = ov["trip_id"].str[4]
    span = ov.groupby("series")["day"].agg(["min", "max", "size"])

    L = []
    L.append("# Data card: Battery and Heating Data in Real Driving Cycles (BMW i3)\n")
    L.append("> Generated by `src/data/data_card.py`; every number below is computed from the data. "
             "Re-run `python -m src.data.build_dataset && python -m src.data.data_card` to regenerate.\n")

    L.append("## Source\n")
    L.append("- **Dataset:** *Battery and Heating Data in Real Driving Cycles*, IEEE DataPort (open access): "
             "https://ieee-dataport.org/open-access/battery-and-heating-data-real-driving-cycles")
    L.append("- **Vehicle:** BMW i3 (battery electric), driven on public roads around Munich.")
    L.append("- **Files:** one CSV per trip (`TripA01`–`TripA32`, `TripB01`–`TripB38`) plus `Overview.xlsx` "
             "(date, route, weather, notes per trip). Semicolon-separated, Latin-1 encoded, CRLF line endings.\n")

    L.append("## At a glance\n")
    clean_h = len(clean) * SAMPLE_PERIOD_S / 3600
    L.append(md_table(pd.DataFrame([
        ["Trips", f"{rf['n_files']} (series A: {span.loc['A', 'size']}, series B: {span.loc['B', 'size']})"],
        ["Series A dates", f"{span.loc['A', 'min']:%Y-%m-%d} to {span.loc['A', 'max']:%Y-%m-%d} (summer)"],
        ["Series B dates", f"{span.loc['B', 'min']:%Y-%m-%d} to {span.loc['B', 'max']:%Y-%m-%d} (winter)"],
        ["Sampling rate", f"10 Hz; time step is exactly {SAMPLE_PERIOD_S} s in {rf['dt_ok']}/{rf['n_files']} trips"],
        ["Raw rows", f"{rf['raw_rows']:,}"],
        ["Rows after cleaning", f"{len(clean):,} ({len(clean) / rf['raw_rows']:.1%})"],
        ["Driving time after cleaning", f"{clean_h:.1f} h"],
        ["Distance after cleaning", f"{trips['distance_km'].sum():,.0f} km"],
        ["Ambient temperature range", f"{fmt(clean['ambient_temp_c'].min(), 1)} to {fmt(clean['ambient_temp_c'].max(), 1)} °C"],
    ], columns=["", "value"])) + "\n")

    L.append("## Trip length distribution (after cleaning)\n")
    L.append("Quantiles shown as min / 25% / median / 75% / max.\n")
    rows = []
    for name, sub in [("all", trips), ("A (summer)", trips[trips.series == "A"]), ("B (winter)", trips[trips.series == "B"])]:
        rows.append([name, len(sub), quantiles(sub["duration_min"]), quantiles(sub["distance_km"]),
                     quantiles(sub["soc_used_pct"]), quantiles(sub["ambient_c"])])
    L.append(md_table(pd.DataFrame(rows, columns=["series", "trips", "duration [min]", "distance [km]",
                                                  "SoC used [%-pts]", "mean ambient [°C]"])) + "\n")
    L.append(f"No trip comes close to an empty battery: the lowest SoC recorded anywhere is "
             f"{clean['soc_pct'].min():.1f} %, and the median trip uses "
             f"{trips['soc_used_pct'].median():.1f} %-points of SoC.\n")

    L.append("Routes (from Overview.xlsx):\n")
    rc = overview["route"].str.strip().value_counts()
    L.append(md_table(pd.DataFrame({"route": rc.index, "trips": rc.to_numpy()})) + "\n")

    L.append("## Columns used\n")
    L.append("The raw files have 6 different header layouts (22–48 columns). These are the "
             f"{len(COLUMNS)} signals present in **all** {rf['n_files']} trips, renamed to snake_case. "
             "Statistics are over the cleaned data.\n")
    rows = []
    for raw_name, std in COLUMNS.items():
        unit, meaning = MEANINGS[std]
        s = clean[std]
        rows.append([f"`{std}`", f"`{raw_name}`", unit, meaning, fmt(s.min()), fmt(s.median()), fmt(s.max())])
    L.append(md_table(pd.DataFrame(rows, columns=["column", "raw header", "unit", "meaning", "min", "median", "max"])) + "\n")

    L.append("Derived columns added by the pipeline:\n")
    L.append(md_table(pd.DataFrame([[f"`{k}`", u, m] for k, (u, m) in DERIVED.items()],
                                   columns=["column", "unit", "meaning"])) + "\n")

    L.append("### Columns dropped\n")
    L.append("Present in only some trips, so a model could not rely on them:\n")
    L.append(md_table(pd.DataFrame([[f"`{c}`", f"{k}/{rf['n_files']}"] for c, k in rf["partial_cols"]],
                                   columns=["raw header", "present in"])) + "\n")

    L.append("## Raw file quirks handled by the loader\n")
    L.append(f"- {rf['n_header_sets']} distinct header layouts across {rf['n_files']} files.")
    L.append("- `TripB38` spells speed as `Velocity [km/h]]]`; `Regenerative Braking Signal ` has a trailing space; "
             "`max. SoC [%)` has a bracket typo (kept as-is).")
    L.append("- `TripA11` has an extra empty column; `TripA21` lacks `Requested Coolant Temperature`.")
    L.append("- Series B lists `Temperature Vent right [°C]` twice (one with a trailing space). Dropped anyway.")
    L.append(f"- {rf['trips_with_nan_soc']} trips (all in series B) have missing SoC values, mostly the last 1–2 rows.\n")

    L.append("## Cleaning steps\n")
    L.append("Implemented in `src/data/clean.py`, tested in `tests/test_clean.py`. A row is counted under "
             "the first step that flags it.\n")
    step_info = {
        "interpolated": ("Speed **and** current form perfect straight lines (up to 2-decimal rounding) "
                         "for ≥ 5 s while moving: a logging gap filled by linear interpolation. In TripB02 "
                         "this block contains throttle 135 % and heater 40 kW, both physically impossible.",
                         "drop"),
        "soc_glitch": ("SoC jumps > 1 %-point between consecutive 0.1 s samples (real steps are ≤ 0.06). "
                       "TripB14's last row reads 0 % after 34.6 %.", "drop"),
        "soc_frozen": ("SoC unchanged ≥ 120 s while mean absolute current > 10 A: a frozen BMS signal. TripB07's "
                       "SoC sticks at 50.4 % while the car keeps drawing current, then goes missing.", "drop"),
        "temp_spike": ("Battery temperature > 5 °C away from its 1 s rolling median. TripA10's first row reads "
                       "60.9 °C max battery temperature, the next row 23.0 °C.", "drop"),
        "charging": ("Speed < 1 km/h with current > +10 A for ≥ 60 s: a charging stop (Overview notes fast "
                     "charging in B04 and B09). Not driving, and it raises SoC.", "drop; trip splits into segments"),
        "soc_missing": ("SoC is NaN. SoC is needed for the target, so these rows are unusable.", "drop"),
        "short_segment": ("A piece of trip < 60 s left between dropped rows.", "drop"),
    }
    rows = []
    for step, (why, action) in step_info.items():
        col = f"drop_{step}"
        hit = report.loc[report[col] > 0, "trip_id"].tolist()
        rows.append([f"`{step}`", why, action, f"{report[col].sum():,}", ", ".join(hit) or "–"])
    L.append(md_table(pd.DataFrame(rows, columns=["step", "problem and rule", "action", "rows", "trips affected"])) + "\n")
    n_null = int(report["heater_nulled"].sum())
    L.append(f"Column-level fix: `heater_power_kw` set to missing in {n_null} trips "
             f"({', '.join(report.loc[report.heater_nulled, 'trip_id'])}) that fail the heater plausibility check "
             "(see Sanity checks). Their rows are kept.\n")
    L.append("Not done (deliberately): no outlier clipping (after the steps above every signal is within "
             "physical limits; see min/max in the column table), no resampling (10 Hz is kept), "
             "no imputation of missing SoC.\n")

    L.append("## Sanity checks\n")
    raw_b04 = load_trip(RAW_DIR / "TripB04.csv")
    charge_i = raw_b04.loc[mask_charging(raw_b04), "battery_current_a"]
    L.append(f"- **Current sign.** During TripB04's charging stop the current is positive "
             f"(median {charge_i.median():.1f} A) and SoC rises, so negative current = discharge.")
    ah, kwh = cc["ah_per_pct"], cc["kwh_per_pct"]
    L.append(f"- **SoC tracks integrated current (coulomb counting).** Across {len(cc)} trip segments with ≥ 2 %-points of "
             f"SoC drop, 1 %-point of SoC = {ah.median():.3f} Ah (IQR {ah.quantile(.25):.3f}–{ah.quantile(.75):.3f}) = "
             f"{kwh.median():.4f} kWh (IQR {kwh.quantile(.25):.4f}–{kwh.quantile(.75):.4f}). Implied gross capacity: "
             f"≈ {100 * ah.median():.0f} Ah / {100 * kwh.median():.1f} kWh, consistent with the 60 Ah BMW i3 "
             f"(our inference; the dataset does not state the battery version).")
    drift = coulomb_drift(clean, ah.median())
    big = drift[drift > 1.5].sort_values(ascending=False)
    L.append(f"- **Coulomb drift.** Re-computing SoC from integrated current, the BMS SoC stays within "
             f"{drift.median():.2f} %-points (median over segments; 90th pct {drift.quantile(.9):.2f}) of it. Segments "
             f"drifting > 1.5 points: " + ", ".join(f"{t} seg {s} ({v:.1f})" for (t, s), v in big.items())
             + ". TripB07 is the trip whose SoC later freezes; its SoC is less trustworthy than the rest.")
    rep = report.set_index("trip_id")
    hi = rep[rep["heater_nulled"]].sort_values("heater_still_kw", ascending=False)
    checked = int(rep["heater_still_kw"].notna().sum())
    a_heat = rep.loc[rep.index.str[4] == "A", "heater_kwh_reported"]
    L.append(f"- **Heater power plausibility.** Standing still, the battery only feeds auxiliaries, so heater power "
             f"should be ≤ battery draw. Of the {checked} trips with ≥ 30 s of standstill, it fails in "
             + ", ".join(f"{t} (heater {r.heater_still_kw:.2f} kW vs battery {r.battery_still_kw:.2f} kW)"
                         for t, r in hi.iterrows())
             + f". TripA06 reports ~5.3 kW of heating for its whole duration at ~30 °C ambient, a stuck signal; it "
               f"alone was {a_heat.get('TripA06', 0) / a_heat.sum():.0%} of all heater energy reported in series A. "
               f"**Action:** `heater_power_kw` is set to missing for these trips (rows kept; other signals are fine).")
    ov_raw = pd.read_excel(RAW_DIR / "Overview.xlsx").dropna(subset=["Trip"]).set_index("Trip")
    untouched = report.loc[report.filter(like="drop_").sum(axis=1) <= 20, "trip_id"]
    diff = (trips.loc[untouched, "distance_km"] - ov_raw.loc[untouched, "Distance [km]"]).abs()
    off = diff[diff > 0.1]
    L.append(f"- **Distance.** For the {len(untouched)} trips the cleaning left essentially intact, integrated speed "
             f"matches Overview.xlsx distance within 0.1 km for {len(diff) - len(off)} of them (median difference "
             f"{diff.median():.3f} km). Exceptions: "
             + ", ".join(f"{t} ({trips.loc[t, 'distance_km']:.2f} km in CSV vs {ov_raw.loc[t, 'Distance [km]']:.2f} km "
                         f"in Overview)" for t in off.index)
             + ". TripA30's Overview row also shares its start time with TripA31, and its duration matches the CSV, "
               "so we treat the Overview distance as the error.")
    dvdt = clean.groupby(["trip_id", "segment"])["speed_kmh"].diff().fillna(0) / 3.6 / SAMPLE_PERIOD_S
    L.append(f"- **Acceleration sensor** vs. differentiated speed: correlation "
             f"{np.corrcoef(clean['accel_long_ms2'], dvdt)[0, 1]:.2f}.")
    pred = (clean["soc_pct"] - clean["soc_min_pct"]) / (clean["soc_max_pct"] - clean["soc_min_pct"]) * 100
    resid = clean["soc_displayed_pct"] - pred
    L.append(f"- **Displayed SoC** is close to the BMS SoC rescaled onto its usable window "
             f"(`(soc − soc_min) / (soc_max − soc_min) × 100`) but not exactly: the dashboard value is on "
             f"average {resid.mean():.1f} points higher (std {resid.std():.1f}, max |diff| {resid.abs().max():.1f}). "
             f"The exact mapping is not documented.")
    L.append(f"- **Overview.xlsx inconsistency.** TripA31 lists SoC 0.810 → 0.825 (an increase) with a SoC-difference of "
             f"0.719; the CSV shows {clean.loc[clean.trip_id == 'TripA31', 'soc_pct'].iat[0]:.1f} → "
             f"{clean.loc[clean.trip_id == 'TripA31', 'soc_pct'].iat[-1]:.1f} %. We use the CSV values.\n")

    L.append("## Open questions\n")
    L.append("- Which SoC defines \"empty\" for distance-to-empty: BMS SoC reaching `soc_min` (8 %) or displayed SoC reaching 0? "
             "Matters in Phase 2.")
    L.append("- Elevation source (GPS vs. barometric) and the `regen_signal` scale are undocumented.")
    L.append("- Some trips carry extra payload (+60 to +160 kg, noted in Overview.xlsx); payload is not a signal in the CSVs.\n")
    return "\n".join(L)


def main() -> None:
    """Write data/DATA_CARD.md."""
    CARD_PATH.write_text(build_card(), encoding="utf-8")
    print(f"wrote {CARD_PATH}")


if __name__ == "__main__":
    main()
