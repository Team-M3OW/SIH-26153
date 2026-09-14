"""The actual working prototype the PS asks for: accepts network traffic
input (a CSV of flow features, or a raw PCAP) through a real HTTP API,
runs the world model LIVE on whatever is uploaded, and serves a dashboard
that displays the freshly-computed forecast + MITRE stage + SHAP
explanation -- not a replay of one precomputed day (see dashboard.html for
that; this is the tool dashboard.html should have been from the start).

Two endpoints:
  POST /api/analyze  -- upload a file, get back the full per-window trace
                        (stage probs + k=1..5 infiltration forecast), no
                        SHAP yet (that's ~100 model evaluations per window
                        via KernelExplainer -- too slow to do eagerly for
                        every window of a real upload).
  POST /api/explain   -- given a session + window index, compute SHAP for
                        JUST that window, on demand (what the frontend
                        calls when the user scrubs to a new window).

Session state (the per-window raw feature rows, needed to reconstruct a
window's SHAP explanation later) lives in an in-memory dict, fine for a
local single-user demo -- not meant to survive a server restart.
"""
from __future__ import annotations

import io
import os
import sys
import uuid

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "wm"))
from live_pipeline import (load_model, explain_current_window, process_packet,  # noqa: E402
                          finalize_window, CONTEXT, K_MAX)
from heads import rollout_transition_matrix, INFIL_IDX  # noqa: E402
from features import INPUT_FEATURE_COLUMNS  # noqa: E402
from attack_map import STAGES  # noqa: E402

import torch  # noqa: E402

app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, HEAD, INFIL_HEAD, PI, SCALER = load_model(DEVICE)
SESSIONS: dict[str, dict] = {}   # session_id -> {"rows": [np.ndarray, ...], "labels": [str|None, ...]}

SAMPLES = {
    "wed-infiltration": ("Wednesday-28-02-2018 (real CIC-IDS-2018 Infiltration attack)",
                         "/media/kavinder/hdd2/sih26153-processed/cic_ids2018/Wednesday-28-02-2018.features.csv"),
    "thu-infiltration": ("Thursday-01-03-2018 (real CIC-IDS-2018 Infiltration attack)",
                         "/media/kavinder/hdd2/sih26153-processed/cic_ids2018/Thursday-01-03-2018.features.csv"),
    "demo-pcap": ("demo_attack.pcap (synthetic packet capture)",
                 "/media/kavinder/hdd2/cyber-world-model/data/demo_attack.pcap"),
}

print(f"model loaded on {DEVICE}")


def rows_from_csv(raw_bytes: bytes):
    df = pd.read_csv(io.BytesIO(raw_bytes))
    missing = [c for c in INPUT_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required feature columns: {missing[:5]}"
                        + (f" (+{len(missing)-5} more)" if len(missing) > 5 else ""))
    rows = [r[INPUT_FEATURE_COLUMNS].values.astype(np.float32) for _, r in df.iterrows()]
    labels = df["stage"].tolist() if "stage" in df.columns else [None] * len(df)
    return rows, labels


def rows_from_pcap(raw_bytes: bytes):
    from scapy.utils import PcapReader
    from scapy.layers.inet import IP
    windows: dict[int, dict] = {}
    retrans_count: dict[int, int] = {}
    day_start = [None]
    reader = PcapReader(io.BytesIO(raw_bytes))
    for pkt in reader:
        t = float(pkt.time)
        ip = IP(bytes(pkt))
        process_packet(ip, windows, day_start, retrans_count, t)
    rows, labels = [], []
    for wid in sorted(windows.keys()):
        row = finalize_window(windows[wid], retrans_count.get(wid, 0))
        if row is not None:
            rows.append(row); labels.append(None)
    return rows, labels


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "demo_live.html")


@app.route("/api/samples")
def samples():
    return jsonify({key: label for key, (label, _) in SAMPLES.items()})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    sample_key = request.form.get("sample")
    if sample_key:
        if sample_key not in SAMPLES:
            return jsonify({"error": f"unknown sample '{sample_key}'"}), 400
        display_name, path = SAMPLES[sample_key]
        with open(path, "rb") as fh:
            raw = fh.read()
        kind = "pcap" if path.lower().endswith((".pcap", ".pcapng")) else "csv"
    else:
        f = request.files.get("file")
        if f is None:
            return jsonify({"error": "no file uploaded"}), 400
        raw, display_name = f.read(), f.filename
        kind = "pcap" if display_name.lower().endswith((".pcap", ".pcapng")) else "csv"
    try:
        rows, labels = rows_from_pcap(raw) if kind == "pcap" else rows_from_csv(raw)
    except Exception as e:
        return jsonify({"error": f"failed to process {kind} file: {e}"}), 400
    if len(rows) < CONTEXT:
        return jsonify({"error": f"only {len(rows)} windows extracted -- need at least "
                                 f"{CONTEXT} (the model's context length) to produce a forecast"}), 400

    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = {"rows": rows, "labels": labels}

    windows_out = []
    context_buf: list[np.ndarray] = []
    for i, row in enumerate(rows):
        context_buf.append(row)
        if len(context_buf) > CONTEXT:
            context_buf.pop(0)
        if len(context_buf) < CONTEXT:
            continue
        ctx_raw = np.stack(context_buf)
        ctx_scaled = SCALER.transform(ctx_raw)
        with torch.no_grad():
            t = torch.from_numpy(ctx_scaled).unsqueeze(0).to(DEVICE)
            z = MODEL.encode(t)
            gamma_t = torch.softmax(HEAD(z[:, -1]), dim=-1).squeeze(0).cpu().numpy()
            out = MODEL.rollout(t, k=K_MAX, n_traj=64)
            lat = out["latents"].squeeze(0)
            infil_neural = torch.sigmoid(INFIL_HEAD(lat)).mean(dim=1).cpu().numpy()
        infil_pi = np.array([rollout_transition_matrix(PI, gamma_t, k)[INFIL_IDX.numpy()].sum()
                             for k in range(1, K_MAX + 1)])
        windows_out.append({
            "idx": i,
            "ground_truth": labels[i],
            "predicted_stage": STAGES[int(gamma_t.argmax())],
            "stage_probs": [float(x) for x in gamma_t],
            "infil_forecast": [float(x) for x in infil_neural],
            "infil_pi": [float(x) for x in infil_pi],
        })

    return jsonify({
        "session_id": session_id,
        "source": display_name,
        "kind": kind,
        "n_raw_windows": len(rows),
        "context": CONTEXT, "k_max": K_MAX, "stages": STAGES,
        "has_ground_truth": any(l is not None for l in labels),
        "windows": windows_out,
    })


@app.route("/api/explain", methods=["POST"])
def explain():
    body = request.get_json(force=True)
    session_id, idx = body.get("session_id"), body.get("idx")
    sess = SESSIONS.get(session_id)
    if sess is None:
        return jsonify({"error": "unknown or expired session -- re-upload the file"}), 404
    rows = sess["rows"]
    if idx < CONTEXT - 1 or idx >= len(rows):
        return jsonify({"error": "index out of range for this session"}), 400
    ctx_raw = np.stack(rows[idx - CONTEXT + 1: idx + 1])
    stage_name, top_features = explain_current_window(MODEL, HEAD, SCALER, ctx_raw, DEVICE)
    return jsonify({
        "predicted_stage": stage_name,
        "shap": [{"feature": n, "value": v, "raw": r} for n, v, r in top_features],
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8792, debug=False)
