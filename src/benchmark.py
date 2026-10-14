"""Phase 5 deployment benchmark: accuracy, latency, size and offline capability per option.

What is measured where (be explicit -- no Android app has been built yet):
- TFLite inference latency: Python TFLite interpreter on THIS computer, 1 thread, batch 1.
  This is a proxy, NOT a phone measurement.
- Server latency: round trip over localhost to a real uvicorn process, so it excludes any
  mobile-network delay. `server_ms` = feature computation + model time inside the server.
- Accuracy: test-set metrics from experiments/phase5_export.csv (run export_tflite first).
- APK size increase and on-device latency: not measured (no app yet).

Run with:  .venv/bin/python -m src.benchmark   -> experiments/phase5_benchmark.csv
"""

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from src.export_tflite import EXPORT_RESULTS, MODEL_DIR, TFLITE_DIR  # noqa: E402
from src.features.online import request_from_clean  # noqa: E402
from src.features.sequences import HEATER_CH  # noqa: E402
from src.models.run_cv import EXP_DIR, load_data  # noqa: E402

BENCH_PATH = EXP_DIR / "phase5_benchmark.csv"
PORT = 8765
N_INVOKES = 2000
N_REQUESTS = 200
SEED = 0


def tflite_latency_ms(path: Path, inputs: list[np.ndarray]) -> tuple[float, float]:
    """Median and 95th-percentile latency (ms) of single invokes, cycling through real test inputs."""
    it = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    it.allocate_tensors()
    pairs = []
    for det in it.get_input_details():
        arr = next(x for x in inputs if tuple(x.shape[1:]) == tuple(det["shape"][1:]))
        pairs.append((det["index"], arr.astype(np.float32)))
    n = len(inputs[0])
    times = []
    for k in range(N_INVOKES + 50):
        for idx, arr in pairs:
            it.set_tensor(idx, arr[k % n: k % n + 1])
        t0 = time.perf_counter()
        it.invoke()
        if k >= 50:  # skip warm-up
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


def server_latency(requests: list[dict]) -> dict:
    """Start uvicorn, send the requests sequentially over keep-alive HTTP, return timing stats."""
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app", "--port", str(PORT), "--log-level", "warning"],
                            cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = f"http://127.0.0.1:{PORT}"
        with httpx.Client(timeout=10) as c:
            for _ in range(100):
                try:
                    if c.get(f"{url}/health").status_code == 200:
                        break
                except httpx.TransportError:
                    time.sleep(0.2)
            for r in requests[:10]:  # warm-up
                c.post(f"{url}/predict", json=r)
            rtt, srv, size = [], [], []
            for r in requests:
                body = json.dumps(r).encode()
                t0 = time.perf_counter()
                resp = c.post(f"{url}/predict", content=body, headers={"content-type": "application/json"})
                rtt.append((time.perf_counter() - t0) * 1000)
                resp.raise_for_status()
                srv.append(resp.json()["server_ms"])
                size.append(len(body) / 1024)
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    return {"rtt_ms_median": float(np.median(rtt)), "rtt_ms_p95": float(np.percentile(rtt, 95)),
            "server_ms_median": float(np.median(srv)), "request_kb_median": float(np.median(size))}


def main() -> None:
    """Measure every deployment option and write the benchmark table."""
    data = load_data()
    a = data["anchors"]
    te = (a["split"] == "test").to_numpy()
    meta = json.loads((MODEL_DIR / "gru_meta.json").read_text())
    card = json.loads((TFLITE_DIR / "model_card.json").read_text())
    seq = data["seq"][te].copy()
    seq[:, :, HEATER_CH] = np.nan_to_num(seq[:, :, HEATER_CH], nan=meta["heater_fill"])
    ctx = data["ctx"][te]
    feats = data["X"].loc[te, card["ridge"]["feature_order"]].to_numpy(np.float32)
    acc = pd.read_csv(EXPORT_RESULTS).set_index("model")

    rows = []
    for name, inputs in [("gru_fp32", [seq, ctx]), ("gru_dynamic_int8", [seq, ctx]), ("ridge_fp32", [feats])]:
        med, p95 = tflite_latency_ms(TFLITE_DIR / f"{name}.tflite", inputs)
        rows.append({"option": f"on-device TFLite: {name}", "test_mae_rate": acc.loc[name, "mae_rate"],
                     "test_mae_dte_km": acc.loc[name, "mae_dte_km"],
                     "model_size_kb": (TFLITE_DIR / f"{name}.tflite").stat().st_size / 1024,
                     "inference_ms_median": med, "inference_ms_p95": p95,
                     "round_trip_ms_median": np.nan, "round_trip_ms_p95": np.nan, "request_kb": np.nan,
                     "works_offline": True, "apk_increase": "not measured (no app yet)"})

    pick = np.random.default_rng(SEED).choice(a.index[te], N_REQUESTS, replace=False)
    reqs = [request_from_clean(data["clean"], a.loc[i]) for i in pick]
    s = server_latency(reqs)
    rows.append({"option": "server: FastAPI + Ridge (sklearn)", "test_mae_rate": acc.loc["ridge_sklearn (reference)", "mae_rate"],
                 "test_mae_dte_km": acc.loc["ridge_sklearn (reference)", "mae_dte_km"],
                 "model_size_kb": (MODEL_DIR / "ridge.joblib").stat().st_size / 1024,
                 "inference_ms_median": s["server_ms_median"], "inference_ms_p95": np.nan,
                 "round_trip_ms_median": s["rtt_ms_median"], "round_trip_ms_p95": s["rtt_ms_p95"],
                 "request_kb": s["request_kb_median"], "works_offline": False, "apk_increase": "0 (model not in app)"})
    out = pd.DataFrame(rows)
    out["measured_on"] = f"{platform.system()} {platform.machine()}, Python TFLite interpreter / localhost"
    out.to_csv(BENCH_PATH, index=False)
    pd.set_option("display.width", 250)
    print(out.drop(columns=["measured_on"]).round(3).to_string(index=False))
    print("\nmeasured on:", out["measured_on"].iat[0])


if __name__ == "__main__":
    main()
