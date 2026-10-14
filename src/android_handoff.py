"""Build the Android handoff package (everything the app needs, one step before building it).

Writes models/handoff/ (gitignored; regenerate with this script):
- gru_fp32.tflite, ridge_fp32.tflite, model_card.json   the on-device models and their I/O contract
- replay_trip.csv        the longest TEST trip at 10 Hz, to replay as a simulated live stream
- golden_vectors.json    inputs + expected outputs for anchors of that trip, so the Kotlin code
                         (window building, feature computation, TFLite call, DTE) can be unit-tested
                         against Python numbers

Run with:  .venv/bin/python -m src.android_handoff   (after export_tflite)
"""

import json
import os
import shutil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

from src.export_tflite import MODEL_DIR, TFLITE_DIR, run_tflite  # noqa: E402
from src.features.online import SAMPLE_FIELDS, implied_dte_km  # noqa: E402
from src.features.sequences import HEATER_CH  # noqa: E402
from src.models.run_cv import load_data  # noqa: E402

HANDOFF_DIR = MODEL_DIR / "handoff"
N_GOLDEN = 20


def main() -> None:
    """Write the handoff files and print a short summary."""
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    for f in ["gru_fp32.tflite", "ridge_fp32.tflite", "model_card.json"]:
        shutil.copy(TFLITE_DIR / f, HANDOFF_DIR / f)

    data = load_data()
    a = data["anchors"]
    test = a[a["split"] == "test"]
    trip_id = test.groupby("trip_id").size().idxmax()           # longest test trip, chosen by rule
    trip = data["clean"][data["clean"]["trip_id"] == trip_id]
    trip[["trip_id", "segment", *SAMPLE_FIELDS]].to_csv(HANDOFF_DIR / "replay_trip.csv", index=False)

    card = json.loads((TFLITE_DIR / "model_card.json").read_text())
    meta = json.loads((MODEL_DIR / "gru_meta.json").read_text())
    idx = test.index[test["trip_id"] == trip_id]
    idx = idx[np.linspace(0, len(idx) - 1, N_GOLDEN).astype(int)]
    pos = a.index.get_indexer(idx)
    seq = data["seq"][pos].copy()
    seq[:, :, HEATER_CH] = np.nan_to_num(seq[:, :, HEATER_CH], nan=meta["heater_fill"])
    ctx = data["ctx"][pos]
    feats = data["X"].loc[idx, card["ridge"]["feature_order"]].to_numpy(np.float32)
    gru_out = run_tflite((TFLITE_DIR / "gru_fp32.tflite").read_bytes(), [seq, ctx])
    ridge_out = run_tflite((TFLITE_DIR / "ridge_fp32.tflite").read_bytes(), [feats])

    golden = []
    for k, i in enumerate(idx):
        row = a.loc[i]
        golden.append({
            "replay_row": int(row["row"] - trip.index[0]),   # 0-based row in replay_trip.csv = "now"
            "time_s": float(row["time_s"]), "soc_pct": float(row["soc_pct"]),
            "gru_window": seq[k].round(6).tolist(), "gru_context": ctx[k].round(6).tolist(),
            "ridge_features": feats[k].round(6).tolist(),
            "expected_gru_rate": float(gru_out[k]), "expected_ridge_rate": float(ridge_out[k]),
            "expected_gru_dte_km": implied_dte_km(float(row["soc_pct"]), float(gru_out[k])),
            "expected_ridge_dte_km": implied_dte_km(float(row["soc_pct"]), float(ridge_out[k])),
            "actual_rate_next_5km": float(row["rate_next"]),
        })
    (HANDOFF_DIR / "golden_vectors.json").write_text(json.dumps(
        {"trip_id": trip_id, "tolerance": 1e-4, "vectors": golden}, indent=1))
    print(f"handoff -> {HANDOFF_DIR}: replay trip {trip_id} ({len(trip):,} samples, "
          f"{len(trip) / 600:.1f} min), {len(golden)} golden vectors")
    for f in sorted(HANDOFF_DIR.iterdir()):
        print(f"  {f.name:<22} {f.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
