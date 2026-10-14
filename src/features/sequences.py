"""GRU inputs: a fixed-length window of recent signals plus a few context values per anchor.

Window: the last WINDOW_S = 300 s before the anchor, averaged into STEP_S = 5 s blocks -> 60 steps.
A fixed length is required for the TFLite export (unrolled GRU, see Phase 0 notes).
Anchors with less than 300 s of history (13.6 %) are padded by repeating the segment's first
sample, so every model is evaluated on exactly the same anchors.

Heater power is NaN for the trips whose heater signal failed the plausibility check; it is left
as NaN here and imputed inside GRUModel with the training median. (A first version added a
"heater missing" flag channel instead; since only two trips with anchors have it set, the GRU
used it as a trip identifier and predicted negative rates for TripA05 in CV.)
"""

import numpy as np
import pandas as pd

from src.data.load import SAMPLE_PERIOD_S

WINDOW_S = 300.0
STEP_S = 5.0
N_STEPS = int(WINDOW_S / STEP_S)                     # 60
SAMPLES_PER_STEP = int(round(STEP_S / SAMPLE_PERIOD_S))  # 50

# per-step channels; elevation is relative to the anchor so the model sees the recent profile
SEQ_CHANNELS = ["speed_kmh", "accel_long_ms2", "power_draw_kw", "elev_rel_m", "ambient_temp_c",
                "heater_power_kw", "battery_temp_c"]
HEATER_CH = SEQ_CHANNELS.index("heater_power_kw")
CONTEXT = ["elapsed_min", "km_in_segment", "soc_pct", "so_far_rate"]


def gru_inputs(clean: pd.DataFrame, anchors: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (sequences [n_anchors, N_STEPS, n_channels], context [n_anchors, len(CONTEXT)]) as float32.

    Block k (k = 0..N_STEPS-1, oldest first) averages the SAMPLES_PER_STEP samples ending
    (N_STEPS - k) * SAMPLES_PER_STEP samples before the anchor; the anchor sample itself is excluded.
    """
    clean = clean.reset_index(drop=True)
    seg_key = clean["trip_id"] + "#" + clean["segment"].astype(str)
    a_key = (anchors["trip_id"] + "#" + anchors["segment"].astype(str)).to_numpy()
    n_win = N_STEPS * SAMPLES_PER_STEP
    seqs = np.zeros((len(anchors), N_STEPS, len(SEQ_CHANNELS)), np.float32)
    ctx = np.zeros((len(anchors), len(CONTEXT)), np.float32)

    for key, s in clean.groupby(seg_key, sort=False):
        idx = np.where(a_key == key)[0]
        if len(idx) == 0:
            continue
        sig = np.column_stack([
            s["speed_kmh"], s["accel_long_ms2"], s["power_draw_kw"], s["elevation_m"],
            s["ambient_temp_c"], s["heater_power_kw"], s["battery_temp_c"],
        ]).astype(np.float64)  # heater NaN propagates to whole blocks -> imputed in GRUModel
        padded = np.vstack([np.repeat(sig[:1], n_win, axis=0), sig])   # front edge-padding
        cs = np.vstack([np.zeros((1, sig.shape[1])), np.cumsum(padded, axis=0)])
        i0 = anchors["row"].to_numpy()[idx] - s.index[0] + n_win          # anchor position in `padded`
        starts = i0[:, None] - n_win + np.arange(N_STEPS)[None, :] * SAMPLES_PER_STEP
        block = (cs[starts + SAMPLES_PER_STEP] - cs[starts]) / SAMPLES_PER_STEP   # [a, steps, ch]
        elev_ch = SEQ_CHANNELS.index("elev_rel_m")
        block[:, :, elev_ch] -= padded[i0, elev_ch][:, None]
        seqs[idx] = block

        t = s["time_s"].to_numpy()
        rows = anchors["row"].to_numpy()[idx] - s.index[0]
        ctx[idx, 0] = (t[rows] - t[0]) / 60
    ctx[:, 1] = anchors["km_in_segment"].to_numpy()
    ctx[:, 2] = anchors["soc_pct"].to_numpy()
    ctx[:, 3] = anchors["so_far_rate"].to_numpy()
    return seqs, ctx
