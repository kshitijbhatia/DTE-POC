"""Server tests: request validation, and PARITY with the offline pipeline on real test trips.

The parity test needs the processed data and trained models (run build_dataset, run_test and
export_tflite first); it is skipped if they are missing.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
NEEDED = [ROOT / "models" / "ridge.joblib", ROOT / "models" / "tflite" / "model_card.json",
          ROOT / "data" / "processed" / "trips_clean.parquet"]
pytestmark = pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="needs trained models and processed data")


def request_for_anchor(clean: pd.DataFrame, anchor: pd.Series) -> dict:
    """Same request builder the benchmark uses."""
    from src.features.online import request_from_clean
    return request_from_clean(clean, anchor)


@pytest.fixture(scope="module")
def setup():
    from server.app import app
    from src.models.run_cv import load_data
    data = load_data()
    with TestClient(app) as client:
        yield client, data


def test_health(setup) -> None:
    client, _ = setup
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_rejects_window_shorter_than_1km(setup) -> None:
    client, data = setup
    a = data["anchors"].query("split == 'test'").iloc[0]
    req = request_for_anchor(data["clean"], a)
    req["samples"] = req["samples"][-50:]  # 5 s of data
    r = client.post("/predict", json=req)
    assert r.status_code == 422 and "need >= 1.0 km" in r.text


def test_server_matches_offline_pipeline(setup) -> None:
    import joblib
    client, data = setup
    a = data["anchors"]
    test_idx = a.index[a.split == "test"]
    pick = np.random.default_rng(0).choice(test_idx, 25, replace=False)
    ridge = joblib.load(ROOT / "models" / "ridge.joblib")
    offline = ridge.predict(data["X"].loc[pick])
    online = [client.post("/predict", json=request_for_anchor(data["clean"], a.loc[i])).json() for i in pick]
    assert np.allclose([r["rate_pct_soc_per_km"] for r in online], offline, atol=1e-9)
    dte_offline = (a.loc[pick, "soc_pct"] - 8.0) / np.maximum(offline, 0.05)
    assert np.allclose([r["dte_km"] for r in online], dte_offline, atol=1e-6)
