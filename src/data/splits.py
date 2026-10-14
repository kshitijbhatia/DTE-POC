"""Train/test split and cross-validation folds, grouped by DAY and stratified by SEASON.

Why by day: trips on the same day share weather, battery temperature and sometimes run
back-to-back (TripB15 -> B16 -> B17). Same-day trips differ by a median 1.1 °C in ambient
temperature vs. 3.3 °C for trips on different days, so splitting by trip would let a model
recognise "this day" from temperature. All trips of a day therefore land in the same split.

Why stratified by season: each split gets summer (A) and winter (B) days in proportion, so a
score never depends on a lucky draw of mostly-summer or mostly-winter days.

Run with:  .venv/bin/python -m src.data.splits   -> writes experiments/configs/splits.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load import load_overview

SPLITS_PATH = Path(__file__).resolve().parents[2] / "experiments" / "configs" / "splits.csv"
SEED = 42
TEST_FRAC = 0.2
N_FOLDS = 5


def trip_days() -> pd.DataFrame:
    """One row per trip: trip_id, series ("A"/"B"), day ("YYYY-MM-DD") from Overview.xlsx."""
    ov = load_overview()
    day = ov["date"].astype(str).str[:10].str.replace("_", "-")
    return pd.DataFrame({"trip_id": ov["trip_id"], "series": ov["trip_id"].str[4], "day": day})


def make_splits(trips: pd.DataFrame, test_frac: float = TEST_FRAC, n_folds: int = N_FOLDS,
                seed: int = SEED) -> pd.DataFrame:
    """Assign every trip to the test set or to one of `n_folds` CV folds.

    Within each season: days are shuffled with `seed`; the first round(test_frac * n_days)
    become test days. Remaining days go to CV folds greedily (largest day first, into the
    fold with the fewest trips so far), which keeps fold sizes balanced in trips.

    Returns `trips` with added columns `split` ("test" or "cv") and `fold` (0..n_folds-1, or -1 for test).
    """
    rng = np.random.default_rng(seed)
    day_split, day_fold = {}, {}
    fold_trips = np.zeros(n_folds, int)
    for series in sorted(trips["series"].unique()):
        days = trips.loc[trips["series"] == series].groupby("day").size()
        order = rng.permutation(days.index.to_numpy())
        n_test = int(round(test_frac * len(order)))
        for d in order[:n_test]:
            day_split[d], day_fold[d] = "test", -1
        # greedy: biggest days first, each into the currently smallest fold (stable tie-break)
        cv_days = sorted(order[n_test:], key=lambda d: -days[d])
        for d in cv_days:
            f = int(np.argmin(fold_trips))
            day_split[d], day_fold[d] = "cv", f
            fold_trips[f] += days[d]
    out = trips.copy()
    out["split"] = out["day"].map(day_split)
    out["fold"] = out["day"].map(day_fold).astype(int)
    return out


def load_splits() -> pd.DataFrame:
    """Read the saved split assignment (trip_id, series, day, split, fold)."""
    return pd.read_csv(SPLITS_PATH)


def main() -> None:
    """Create the split, save it, and print a summary."""
    s = make_splits(trip_days())
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(SPLITS_PATH, index=False)
    summary = s.assign(group=np.where(s["split"] == "test", "test", "fold " + s["fold"].astype(str)))
    print(summary.pivot_table(index="group", columns="series", values="trip_id", aggfunc="count", fill_value=0)
          .assign(days=summary.groupby("group")["day"].nunique()))
    print(f"wrote {SPLITS_PATH}")


if __name__ == "__main__":
    main()
