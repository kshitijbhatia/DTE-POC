# Android app: handoff spec (app not built yet)

Phase 5 stops one step before the app. Everything the app needs exists and is verified in Python;
this document is the contract for building it.

## What is ready

Generate with `python -m src.export_tflite && python -m src.android_handoff` → `models/handoff/`:

| File | What it is |
|---|---|
| `gru_fp32.tflite` (181 KB) | GRU, batch 1, built-in TFLite ops only (no Flex delegate needed) |
| `ridge_fp32.tflite` (1 KB) | Ridge folded into one dense layer (imputer + scaler + weights) |
| `model_card.json` | input shapes, channel/feature order, DTE rule, baseline rate |
| `replay_trip.csv` | TripB09 (a held-out test trip), 30,817 samples at 10 Hz (51 min) |
| `golden_vectors.json` | 20 points of that trip: exact model inputs + expected outputs (tolerance 1e-4) |

The dynamic-range quantized GRU is not included: it is byte-identical to float32 because TFLite
skips weight tensors under 1,024 elements, and this GRU's largest has 768 (`src/export_tflite.py`).

## Models: inputs and outputs

Both output `rate` = predicted consumption over the next 5 km, in %SoC per km. Then
`dte_km = (soc_pct − 8.0) / max(rate, 0.05)` (8 % = the BMS lower limit). Baseline DTE uses the fixed
rate in `model_card.json` (`baseline_global_avg_rate` ≈ 0.707).

Predictions start once the trip has **≥ 1 km** behind it (the models never saw earlier points).

**GRU**: two float32 inputs.
- `window [1, 60, 7]`: last 300 s before "now", averaged into 60 blocks of 5 s (50 samples each), oldest first.
  Channels: `speed_kmh, accel_long_ms2, power_draw_kw, elev_rel_m, ambient_temp_c, heater_power_kw, battery_temp_c`.
  `power_draw_kw = −voltage × current / 1000` (current is negative when discharging).
  `elev_rel_m` = block mean elevation − elevation now. If the trip is shorter than 300 s, pad the
  missing (oldest) samples by repeating the trip's first sample.
- `context [1, 4]`: `elapsed_min, km_in_segment (km since trip start), soc_pct (now), so_far_rate`
  where `so_far_rate = (soc at trip start − soc now) / km since trip start`.
- Normalisation is inside the model: feed raw units.

**Ridge**: `features [1, 14]` in the order listed in `model_card.json`. Definitions are in
`src/features/build.py`; "1 km" windows cover the samples in the last 1.0 km before "now",
excluding the "now" sample; distance = integrated speed (`speed / 3600 × 0.1` km per sample).

## Verify before trusting the app

For each entry in `golden_vectors.json`: replay `replay_trip.csv` up to row `replay_row`, build the
inputs on-device, compare with `gru_window` / `gru_context` / `ridge_features`, run the model and compare
with `expected_*` (|diff| ≤ 1e-4). This catches channel-order, unit and windowing bugs, which are the
usual source of "the model works in Python but not on the phone".

## Planned app (when resumed)

- Replay `replay_trip.csv` as a `Flow<Sample>` emitting every 100 ms (real 10 Hz), mimicking an MQTT feed.
- Ring buffer of the last 300 s; predict every 10 s once ≥ 1 km is driven (running TFLite every 100 ms is unnecessary).
- Show live DTE: on-device model, baseline, and server (`POST /predict`, see `server/app.py`), with an on-device/server toggle.
- **Send features, not raw samples, to a production server.** The current `/predict` takes the raw
  10 Hz window (≈ 127 KB JSON per request in the benchmark) so the server can reuse the training code
  exactly; that is fine on localhost but heavy on mobile data.

## Still to measure (needs the app)

On-device inference latency (the benchmark's numbers are from a laptop), APK size increase from the
TFLite runtime and model, and server round trip over a real mobile network.
