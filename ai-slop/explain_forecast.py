"""Extends SHAP to explain the k-step-ahead FORECAST, not just the current
window's stage classification -- directly addressing the report's stated
gap: the original explain_current_window() (live_pipeline.py) only
explains "why does the model think THIS window is stage X," never "why
does the model think infiltration probability at t+k is Y." This computes
SHAP against sigmoid(infil_head(rollout(...)))[k], holding the prior
context fixed and perturbing only the current window's 43 features, for a
chosen horizon k.

Rollout inside a KernelExplainer's perturbation loop is expensive (each of
nsamples perturbed inputs needs a full n_traj-sample rollout) -- kept
deliberately smaller (nsamples=60, n_traj=32) than the single-window
explainer, and that tradeoff is stated here rather than hidden.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "wm"))
from features import INPUT_FEATURE_COLUMNS  # noqa: E402
from live_pipeline import load_model, CONTEXT, K_MAX  # noqa: E402


def explain_forecast(model, infil_head, scaler, context_raw: np.ndarray, k: int, device,
                     nsamples=60, n_traj=32):
    """context_raw: (CONTEXT, 43) raw feature rows. k: 0-indexed horizon
    (0 = one window ahead). Returns (baseline_prob_at_k, [(feature, shap_value, raw_value), ...])."""
    import shap

    ctx_scaled_prefix = scaler.transform(context_raw[:-1])
    last_raw = context_raw[-1]

    def predict(X_raw: np.ndarray) -> np.ndarray:
        X_scaled = scaler.transform(X_raw.astype(np.float32))
        seqs = np.stack([np.concatenate([ctx_scaled_prefix, x[None, :]], axis=0) for x in X_scaled])
        with torch.no_grad():
            t = torch.from_numpy(seqs).to(device)
            out = model.rollout(t, k=K_MAX, n_traj=n_traj)
            lat = out["latents"][:, k]                     # (N, n_traj, d_latent) at horizon k
            probs = torch.sigmoid(infil_head(lat)).mean(dim=1).cpu().numpy()
        return probs.reshape(-1, 1)

    background = last_raw[None, :] + np.random.randn(30, len(last_raw)) * (np.abs(last_raw) * 0.1 + 1e-3)
    explainer = shap.KernelExplainer(predict, background)
    shap_values = explainer.shap_values(last_raw[None, :], nsamples=nsamples, silent=True)
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values
    sv = np.asarray(sv).reshape(-1)
    baseline_prob = float(predict(last_raw[None, :])[0, 0])
    order = np.argsort(-np.abs(sv))[:8]
    return baseline_prob, [(INPUT_FEATURE_COLUMNS[i], float(sv[i]), float(last_raw[i])) for i in order]


def main():
    """Demo: compare single-window SHAP (current stage) vs multi-step SHAP
    (k=5-ahead infiltration forecast) on the real Wed-28-02-2018 attack,
    at the exact window this session already validated the live pipeline
    against."""
    import pandas as pd
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, head, infil_head, pi, scaler = load_model(device)

    df = pd.read_csv("/media/kavinder/hdd2/sih26153-processed/cic_ids2018/Wednesday-28-02-2018.features.csv")
    t = 44   # inside the validated Lateral Movement attack window, ground_truth already confirmed
    ctx_raw = df[INPUT_FEATURE_COLUMNS].values.astype(np.float32)[t - CONTEXT:t]

    from live_pipeline import explain_current_window
    stage_name, top_single = explain_current_window(model, head, scaler, ctx_raw, device)
    print(f"single-window SHAP (explains: current stage = '{stage_name}'):")
    for name, val, raw in top_single:
        print(f"    {name:20s} shap={val:+.4f}  raw={raw:.3f}")

    print()
    for k in [0, 4]:   # k=1 and k=5 ahead
        baseline, top_multi = explain_forecast(model, infil_head, scaler, ctx_raw, k, device)
        print(f"multi-step SHAP (explains: infiltration prob at k={k+1}, value={baseline:.3f}):")
        for name, val, raw in top_multi:
            print(f"    {name:20s} shap={val:+.4f}  raw={raw:.3f}")
        print()


if __name__ == "__main__":
    main()
