"""Tests for src.data.load using tiny hand-written CSVs (no real data needed)."""

from pathlib import Path

import pytest

from src.data.load import COLUMNS, load_trip

RAW_HEADERS = list(COLUMNS)


def _write_csv(path: Path, headers: list[str], n_rows: int = 3) -> Path:
    lines = [";".join(headers)]
    for i in range(n_rows):
        lines.append(";".join(str(i * 0.1) if j == 0 else "1.5" for j in range(len(headers))))
    path.write_bytes("\r\n".join(lines).encode("latin-1"))  # same encoding/line endings as raw data
    return path


def test_load_trip_standardises_columns(tmp_path: Path) -> None:
    df = load_trip(_write_csv(tmp_path / "TripB07.csv", RAW_HEADERS + ["Extra B-only column [°C]"]))
    assert list(df.columns) == ["trip_id", "series"] + list(COLUMNS.values())
    assert df["trip_id"].iat[0] == "TripB07" and df["series"].iat[0] == "B"
    assert len(df) == 3 and df["time_s"].tolist() == pytest.approx([0.0, 0.1, 0.2])


def test_load_trip_handles_known_header_typos(tmp_path: Path) -> None:
    headers = [
        "Velocity [km/h]]]" if h == "Velocity [km/h]" else h + "  " if h.startswith("Regen") else h
        for h in RAW_HEADERS
    ]
    df = load_trip(_write_csv(tmp_path / "TripA01.csv", headers))
    assert "speed_kmh" in df.columns and "regen_signal" in df.columns


def test_load_trip_raises_on_missing_column(tmp_path: Path) -> None:
    headers = [h for h in RAW_HEADERS if h != "SoC [%]"]
    with pytest.raises(KeyError, match="SoC"):
        load_trip(_write_csv(tmp_path / "TripA02.csv", headers))
