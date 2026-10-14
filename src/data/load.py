"""Load the BMW i3 "Battery and Heating Data in Real Driving Cycles" trips.

Raw files are semicolon-separated, Latin-1 encoded CSVs (one per trip) sampled at 10 Hz.
Headers differ slightly between files (6 distinct header sets across 70 trips), so every
trip is mapped onto one common schema (`COLUMNS`) holding only the signals present in all
trips. Series-B-only signals (cabin/vent/defrost temperatures, coolant flow, ...) are dropped:
they exist only for the winter trips, so a model could not use them on summer trips.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "bmw_i3"
SAMPLE_PERIOD_S = 0.1  # 10 Hz, verified for every trip in the data card

# Raw header -> standard name. Raw headers are first normalised by `_normalise_header`
# (strip whitespace, fix the "Velocity [km/h]]]" typo in TripB38).
COLUMNS: dict[str, str] = {
    "Time [s]": "time_s",
    "Velocity [km/h]": "speed_kmh",
    "Elevation [m]": "elevation_m",
    "Throttle [%]": "throttle_pct",
    "Motor Torque [Nm]": "motor_torque_nm",
    "Longitudinal Acceleration [m/s^2]": "accel_long_ms2",
    "Regenerative Braking Signal": "regen_signal",
    "Battery Voltage [V]": "battery_voltage_v",
    "Battery Current [A]": "battery_current_a",
    "Battery Temperature [°C]": "battery_temp_c",
    "max. Battery Temperature [°C]": "battery_temp_max_c",
    "SoC [%]": "soc_pct",
    "displayed SoC [%]": "soc_displayed_pct",
    "min. SoC [%]": "soc_min_pct",
    "max. SoC [%)": "soc_max_pct",  # sic: unit bracket typo in the raw header
    "Heating Power CAN [kW]": "heater_power_kw",
    "Requested Heating Power [W]": "heater_requested_w",
    "AirCon Power [kW]": "aircon_power_kw",
    "Heater Signal": "heater_signal",
    "Ambient Temperature [°C]": "ambient_temp_c",
    "Heat Exchanger Temperature [°C]": "heat_exchanger_temp_c",
    "Cabin Temperature Sensor [°C]": "cabin_temp_c",
}


def _normalise_header(name: str) -> str:
    """Strip whitespace and fix known typos in a raw column header."""
    return name.strip().replace("]]]", "]")


def trip_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Return the sorted list of trip CSV paths (TripA01.csv ... TripB38.csv)."""
    files = sorted(raw_dir.glob("Trip*.csv"))
    if not files:
        raise FileNotFoundError(f"No Trip*.csv files in {raw_dir}. See README for how to get the data.")
    return files


def load_trip(path: Path) -> pd.DataFrame:
    """Load one trip CSV into the standard schema.

    Adds `trip_id` (e.g. "TripA01") and `series` ("A" = summer 2019, "B" = winter 2019/20).
    Raises KeyError if a standard column is missing, so header drift is caught early.
    """
    df = pd.read_csv(path, sep=";", encoding="latin-1")
    df.columns = [_normalise_header(c) for c in df.columns]
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name}: missing columns {missing}")
    df = df[list(COLUMNS)].rename(columns=COLUMNS).astype("float64")
    df.insert(0, "trip_id", path.stem)
    df.insert(1, "series", path.stem[4])
    return df


def load_all(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load every trip and concatenate into one long DataFrame (rows ordered by trip, then time)."""
    return pd.concat([load_trip(p) for p in trip_files(raw_dir)], ignore_index=True)


def load_overview(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load the per-trip metadata sheet (date, route, weather, notes) from Overview.xlsx.

    Only descriptive columns are kept; numeric summaries (distance, SoC start/end) are
    recomputed from the CSVs instead, because the sheet has at least one inconsistent row
    (TripA31 lists a SoC *increase* that the CSV does not show).
    """
    ov = pd.read_excel(raw_dir / "Overview.xlsx")
    ov = ov[ov["Trip"].astype(str).str.startswith("Trip")]
    ov = ov.rename(columns={"Trip": "trip_id", "Date": "date", "Route/Area": "route",
                            "Weather": "weather", "Note": "note"})
    return ov[["trip_id", "date", "route", "weather", "note"]].reset_index(drop=True)
