"""Runs the real attack-day replay (see live_pipeline.py --source csv) and
dumps a structured JSON trace -- full stage-probability distribution,
k=1..5 infiltration forecast (both neural and Pi), and top-8 SHAP features
per window -- for the frontend dashboard to animate through, since the
actual model inference has to happen here in Python (a static artifact
can't run PyTorch), not live in the browser.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "wm"))
from live_pipeline import load_model, explain_current_window, CONTEXT, K_MAX  # noqa: E402
from heads import rollout_transition_matrix, INFIL_IDX  # noqa: E402
from features import INPUT_FEATURE_COLUMNS  # noqa: E402
from attack_map import STAGES  # noqa: E402

CSV_PATH = "/media/kavinder/hdd2/sih26153-processed/cic_ids2018/Wednesday-28-02-2018.features.csv"
OUT_PATH = "/home/kavinder/ARSH_ARNABI/SIH26153/ai-slop/demo_data.json"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, head, infil_head, pi, scaler = load_model(device)
    df = pd.read_csv(CSV_PATH)

    context_buf: list[np.ndarray] = []
    windows_out = []

    for i, r in df.iterrows():
        row = r[INPUT_FEATURE_COLUMNS].values.astype(np.float32)
        context_buf.append(row)
        if len(context_buf) > CONTEXT:
            context_buf.pop(0)
        if len(context_buf) < CONTEXT:
            continue

        ctx_raw = np.stack(context_buf)
        ctx_scaled = scaler.transform(ctx_raw)
        with torch.no_grad():
            t = torch.from_numpy(ctx_scaled).unsqueeze(0).to(device)
            z = model.encode(t)
            gamma_t = torch.softmax(head(z[:, -1]), dim=-1).squeeze(0).cpu().numpy()
            out = model.rollout(t, k=K_MAX, n_traj=64)
            lat = out["latents"].squeeze(0)
            infil_neural = torch.sigmoid(infil_head(lat)).mean(dim=1).cpu().numpy()
        infil_pi = np.array([rollout_transition_matrix(pi, gamma_t, k)[INFIL_IDX.numpy()].sum()
                             for k in range(1, K_MAX + 1)])

        try:
            stage_name, top_features = explain_current_window(model, head, scaler, ctx_raw, device)
            shap_out = [{"feature": name, "value": val, "raw": raw} for name, val, raw in top_features]
        except Exception:
            stage_name, shap_out = STAGES[int(gamma_t.argmax())], []

        windows_out.append({
            "idx": int(i),
            "ground_truth": r["stage"],
            "predicted_stage": stage_name,
            "stage_probs": [float(x) for x in gamma_t],
            "infil_forecast": [float(x) for x in infil_neural],
            "infil_pi": [float(x) for x in infil_pi],
            "shap": shap_out,
            "raw_n_flows": float(r["n_flows"]),
        })
        if i % 50 == 0:
            print(f"  processed window {i}/{len(df)}", flush=True)

    payload = {
        "day": "Wednesday-28-02-2018 (real CIC-IDS-2018 Infiltration attack)",
        "context": CONTEXT, "k_max": K_MAX, "stages": STAGES,
        "windows": windows_out,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f)
    print(f"wrote {len(windows_out)} windows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
