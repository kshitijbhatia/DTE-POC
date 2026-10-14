"""Create ready-to-send example requests for the API, from held-out TEST trips.

Writes server/examples/:
- <name>.json        a complete POST /predict body (telemetry window + trip context)
- expected.json      for each example: what the server should answer (offline Ridge) and what
                     actually happened over the next 5 km, so you can compare model vs reality

Examples are picked by rule, not by hand: the middle prediction point of three test trips --
the coldest-weather, the warmest-weather and the fastest test trip.

Run with:  .venv/bin/python -m src.make_api_examples   (after run_test and export_tflite)
"""

import json
from pathlib import Path

import joblib

from src.features.build import tabular_features
from src.features.online import implied_dte_km, request_from_clean
from src.models.run_cv import load_data

ROOT = Path(__file__).resolve().parents[1]
EX_DIR = ROOT / "server" / "examples"


def main() -> None:
    """Pick three test prediction points, write their request bodies and the expected answers."""
    data = load_data()
    a = data["anchors"]
    test = a[a["split"] == "test"].copy()
    feats = tabular_features(data["clean"], test)
    test["ambient"], test["speed"] = feats["ambient_temp_c"], feats["speed_mean_1km"]
    by_trip = test.groupby("trip_id").agg(ambient=("ambient", "mean"), speed=("speed", "mean"))
    picks = {"cold_winter_trip": by_trip["ambient"].idxmin(),
             "warm_summer_trip": by_trip["ambient"].idxmax(),
             "fast_highway_trip": by_trip["speed"].idxmax()}
    ridge = joblib.load(ROOT / "models" / "ridge.joblib")
    EX_DIR.mkdir(parents=True, exist_ok=True)
    expected = {}
    for name, trip_id in picks.items():
        rows = test[test["trip_id"] == trip_id]
        anchor = rows.iloc[len(rows) // 2]
        req = request_from_clean(data["clean"], anchor)
        (EX_DIR / f"{name}.json").write_text(json.dumps(req))
        rate = float(ridge.predict(data["X"].loc[[anchor.name]])[0])
        expected[name] = {
            "trip_id": trip_id, "km_into_trip": round(float(anchor["km_in_segment"]), 2),
            "soc_now_pct": float(anchor["soc_pct"]), "samples_in_request": len(req["samples"]),
            "expected_server_rate_pct_soc_per_km": round(rate, 4),
            "expected_server_dte_km": round(implied_dte_km(float(anchor["soc_pct"]), rate), 1),
            "actual_rate_next_5km": round(float(anchor["rate_next"]), 4),
            "actual_dte_km_if_next_5km_continued": round(float(anchor["dte_true_km"]), 1),
        }
        print(f"{name:<18} {trip_id}  {len(req['samples'])} samples  -> expected rate {rate:.3f}, "
              f"actual {anchor['rate_next']:.3f}")
    (EX_DIR / "expected.json").write_text(json.dumps(expected, indent=2))
    print(f"wrote {EX_DIR}")


if __name__ == "__main__":
    main()
