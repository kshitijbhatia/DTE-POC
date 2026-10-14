"""Export the deployment models to TFLite and measure what quantization costs.

- GRU (models/gru.keras, trained by run_test): float32 and dynamic-range quantized
  (weights stored as int8, activations float). Rebuilt with batch size 1 and unrolled,
  so the converter only needs TFLite built-in ops (no Flex delegate on Android).
  Finding: dynamic-range quantization leaves this GRU unchanged, because TFLite only quantizes
  weight tensors with >= 1024 elements and this model's largest has fewer (printed at run time).
- Ridge (models/ridge.joblib): the sklearn pipeline (median imputer -> scaler -> ridge) is folded
  into one Dense layer: rate = x . w' + b'. On-device inputs are always complete, so the imputer's
  "missing" indicator columns are folded in as 0.

Every TFLite model is checked against its original on the TEST anchors.

Run with:  .venv/bin/python -m src.export_tflite    -> models/tflite/*.tflite, experiments/phase5_export.csv
"""

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from src.evaluate import metrics  # noqa: E402
from src.features.sequences import CONTEXT, HEATER_CH, N_STEPS, SEQ_CHANNELS  # noqa: E402
from src.models.baselines import GlobalAverage  # noqa: E402
from src.models.gru import build_gru  # noqa: E402
from src.models.run_cv import EXP_DIR, load_data  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
TFLITE_DIR = MODEL_DIR / "tflite"
EXPORT_RESULTS = EXP_DIR / "phase5_export.csv"


def load_gru() -> tuple[tf.keras.Model, tf.keras.Model, dict]:
    """Return (trained GRU, the same GRU rebuilt with batch size 1, metadata).

    Two pitfalls found while exporting (both verified against the trained model):
    - Keras' Normalization layer keeps derived mean/variance tensors that set_weights() does not
      refresh, so finalize_state() must be called after copying weights (else outputs ~1e7).
    - Converting the trained model via a traced tf.function gave NaNs / wrong outputs in TFLite;
      converting a fixed-batch Keras model with from_keras_model (the Phase 0 path) is exact.
    """
    trained = tf.keras.models.load_model(MODEL_DIR / "gru.keras")
    meta = json.loads((MODEL_DIR / "gru_meta.json").read_text())
    dummy_seq = np.zeros((2, N_STEPS, len(SEQ_CHANNELS)), np.float32)
    dummy_ctx = np.zeros((2, len(CONTEXT)), np.float32)
    m = build_gru(N_STEPS, len(SEQ_CHANNELS), len(CONTEXT), meta["hidden"], dummy_seq, dummy_ctx, batch_size=1)
    m.set_weights(trained.get_weights())
    for layer in m.layers:
        if isinstance(layer, tf.keras.layers.Normalization):
            layer.finalize_state()
    return trained, m, meta


def convert(model: tf.keras.Model, quantize: bool) -> bytes:
    """Convert a Keras model to TFLite (built-in ops only), optionally dynamic-range quantized."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if quantize:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    return conv.convert()


def largest_weight_elements(model: tf.keras.Model) -> int:
    """Element count of the largest trainable weight matrix (TFLite dynamic-range quantization
    only quantizes weight tensors with at least 1,024 elements by default)."""
    return int(max(np.prod(w.shape) for w in model.trainable_weights))


def int8_weight_tensors(model_bytes: bytes) -> int:
    """Count int8 tensors in a TFLite model (non-zero = some weights were actually quantized)."""
    it = tf.lite.Interpreter(model_content=model_bytes)
    return sum(d["dtype"] == np.int8 for d in it.get_tensor_details())


def ridge_as_keras(ridge_model) -> tuple[tf.keras.Model, list[str]]:
    """Fold imputer + scaler + ridge into a single Dense layer over the raw feature vector."""
    imputer, scaler, ridge = ridge_model.pipe_.named_steps.values()
    feats = list(ridge_model.features)
    n = len(feats)
    w = ridge.coef_ / scaler.scale_          # weights for [features..., indicators...]
    b = ridge.intercept_ - np.sum(ridge.coef_ * scaler.mean_ / scaler.scale_)
    w_feat = w[:n]                            # indicator columns (beyond n) are 0 on-device -> drop
    inp = tf.keras.Input((n,), batch_size=1, name="features")
    dense = tf.keras.layers.Dense(1, name="rate")
    out = dense(inp)
    model = tf.keras.Model(inp, out)
    dense.set_weights([w_feat.reshape(n, 1).astype(np.float32), np.array([b], np.float32)])
    return model, feats


def run_tflite(model_bytes: bytes, inputs: list[np.ndarray]) -> np.ndarray:
    """Run a TFLite model one sample at a time (batch size 1) and return all outputs.

    Each array in `inputs` is matched to the model input with the same per-sample shape.
    """
    it = tf.lite.Interpreter(model_content=model_bytes, num_threads=1)
    it.allocate_tensors()
    out_idx = it.get_output_details()[0]["index"]
    pairs = []
    for det in it.get_input_details():
        match = [arr for arr in inputs if tuple(arr.shape[1:]) == tuple(det["shape"][1:])]
        assert len(match) == 1, f"cannot match input {det['name']} {det['shape']}"
        pairs.append((det["index"], match[0].astype(np.float32)))
    preds = np.empty(len(inputs[0]), np.float32)
    for k in range(len(inputs[0])):
        for idx, arr in pairs:
            it.set_tensor(idx, arr[k:k + 1])
        it.invoke()
        preds[k] = it.get_tensor(out_idx).ravel()[0]
    return preds


def main() -> None:
    """Export all models, verify them on the test anchors, and write sizes + accuracy."""
    TFLITE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    a = data["anchors"]
    te = (a["split"] == "test").to_numpy()
    y, usable = a.loc[te, "rate_next"].to_numpy(), a.loc[te, "usable_soc_pct"].to_numpy()
    rows = []

    # --- GRU --------------------------------------------------------------------------
    trained, gru, meta = load_gru()
    seq = data["seq"][te].copy()
    seq[:, :, HEATER_CH] = np.where(np.isnan(seq[:, :, HEATER_CH]), meta["heater_fill"], seq[:, :, HEATER_CH])
    ctx = data["ctx"][te]
    keras_pred = trained.predict([seq, ctx], batch_size=256, verbose=0).ravel()
    print(f"GRU largest weight tensor: {largest_weight_elements(gru)} elements (dynamic-range threshold: 1024)")
    rows.append({"model": "gru_keras (reference)", "size_kb": (MODEL_DIR / "gru.keras").stat().st_size / 1024,
                 **metrics(y, keras_pred, usable), "max_abs_diff_vs_reference": 0.0, "int8_tensors": 0})
    for variant, quant in [("gru_fp32", False), ("gru_dynamic_int8", True)]:
        b = convert(gru, quant)
        (TFLITE_DIR / f"{variant}.tflite").write_bytes(b)
        p = run_tflite(b, [ctx, seq])
        rows.append({"model": variant, "size_kb": len(b) / 1024, **metrics(y, p, usable),
                     "max_abs_diff_vs_reference": float(np.max(np.abs(p - keras_pred))),
                     "int8_tensors": int8_weight_tensors(b)})

    # --- Ridge ------------------------------------------------------------------------
    ridge = joblib.load(MODEL_DIR / "ridge.joblib")
    rk, feats = ridge_as_keras(ridge)
    Xte = data["X"].loc[te, feats]
    assert not Xte.isna().any().any(), "test anchors should have complete features"
    sk_pred = ridge.predict(data["X"].loc[te])
    b = convert(rk, quantize=False)
    (TFLITE_DIR / "ridge_fp32.tflite").write_bytes(b)
    p = run_tflite(b, [Xte.to_numpy(np.float32)])
    rows.append({"model": "ridge_sklearn (reference)", "size_kb": (MODEL_DIR / "ridge.joblib").stat().st_size / 1024,
                 **metrics(y, sk_pred, usable), "max_abs_diff_vs_reference": 0.0, "int8_tensors": 0})
    rows.append({"model": "ridge_fp32", "size_kb": len(b) / 1024, **metrics(y, p, usable),
                 "max_abs_diff_vs_reference": float(np.max(np.abs(p - sk_pred))), "int8_tensors": int8_weight_tensors(b)})

    # --- metadata the Android side needs --------------------------------------------------
    (TFLITE_DIR / "model_card.json").write_text(json.dumps({
        "gru": {"inputs": {"window": [1, N_STEPS, len(SEQ_CHANNELS)], "context": [1, len(CONTEXT)]},
                "window_channels": SEQ_CHANNELS, "context_order": CONTEXT,
                "window": "last 300 s before 'now', averaged into 60 blocks of 5 s, oldest first; "
                          "elev_rel_m = block mean elevation minus elevation now",
                "heater_fill_if_missing_kw": meta["heater_fill"], "output": "rate_pct_soc_per_km"},
        "ridge": {"inputs": {"features": [1, len(feats)]}, "feature_order": feats,
                  "output": "rate_pct_soc_per_km"},
        "dte": "dte_km = (soc_pct - 8.0) / max(rate, 0.05)",
        # global-average baseline fitted on all CV (training) trips, as in run_test
        "baseline_global_avg_rate": GlobalAverage().fit(
            data["clean"][data["clean"]["trip_id"].isin(data["splits"].query("split == 'cv'")["trip_id"])]).rate_,
    }, indent=2))

    out = pd.DataFrame(rows)
    out.to_csv(EXPORT_RESULTS, index=False)
    pd.set_option("display.width", 200)
    print(out[["model", "size_kb", "mae_rate", "rmse_rate", "mae_dte_km", "max_abs_diff_vs_reference",
               "int8_tensors"]].round(6).to_string(index=False))


if __name__ == "__main__":
    main()
