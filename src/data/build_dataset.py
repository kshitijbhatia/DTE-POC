"""Build the cleaned dataset: raw CSVs -> data/processed/trips_clean.parquet.

Run with:  .venv/bin/python -m src.data.build_dataset
Also writes data/processed/cleaning_report.csv (rows dropped per trip, per cleaning step).
"""

from pathlib import Path

from src.data.clean import clean_all
from src.data.load import load_all

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
CLEAN_PATH = PROCESSED_DIR / "trips_clean.parquet"
REPORT_PATH = PROCESSED_DIR / "cleaning_report.csv"


def main() -> None:
    """Load all raw trips, clean them, and write the parquet file and the cleaning report."""
    raw = load_all()
    clean, report = clean_all(raw)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(CLEAN_PATH, index=False)
    report.to_csv(REPORT_PATH, index=False)

    drops = report.filter(like="drop_").sum()
    print(f"raw rows:   {len(raw):,}")
    for step, n in drops.items():
        print(f"  {step:<22} {n:>8,}")
    print(f"clean rows: {len(clean):,} ({len(clean) / len(raw):.1%} kept), "
          f"{clean['trip_id'].nunique()} trips, {clean.groupby(['trip_id', 'segment']).ngroups} segments")
    print(f"wrote {CLEAN_PATH.relative_to(PROCESSED_DIR.parents[1])} and {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
