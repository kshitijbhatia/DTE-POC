"""Smoke tests: the environment and raw data are in place."""

from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "bmw_i3"


def test_core_imports() -> None:
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import sklearn  # noqa: F401


def test_raw_data_present() -> None:
    trips = sorted(RAW_DIR.glob("Trip*.csv"))
    assert len(trips) == 70, f"expected 70 trip CSVs in {RAW_DIR}, found {len(trips)}"
