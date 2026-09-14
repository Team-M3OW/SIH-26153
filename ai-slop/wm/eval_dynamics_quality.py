"""Direct measurement of open-loop multi-step DYNAMICS quality (raw
43-feature reconstruction error, in scaled units), separate from any
downstream classification/forecast head -- the metric the report's
"Evaluation Scope" section flags as missing. Compares frozen vs unfrozen
RSSM to answer directly: did unfreezing the encoder for Phase 2 help or
hurt the actual learned dynamics, independent of the two supervised heads
built on top of it?

Uses model.imagine(context, horizon) + model.decode(...) exactly as the
Phase-1 training loss does, but reports the per-horizon MSE/NLL as a
standalone number instead of folding it into a downstream head's loss.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_days, temporal_split, RobustScaler  # noqa: E402
from features import INPUT_FEATURE_COLUMNS  # noqa: E402
from arch_loader import load_world_model  # noqa: E402

CONTEXT, K_MAX = 16, 5
CKPT_DIR = "/media/kavinder/hdd2/sih26153-processed/checkpoints"


@torch.no_grad()
def eval_dynamics(phase1_ckpt: str, device: str, max_windows: int | None = None):
    model, cfg, ckpt1, arch = load_world_model(phase1_ckpt, device)
    scaler = RobustScaler()
    scaler.center = ckpt1["scaler_center"].cpu().numpy()
    scaler.scale = ckpt1["scaler_scale"].cpu().numpy()
    context = ckpt1["context"]

    days = load_days()
    splits = {name: temporal_split(df) for name, df in days.items()}

    sq_err = {k: [] for k in range(K_MAX)}
    n_eval = 0
    for name, s in splits.items():
        df = s["test"]
        if len(df) < context + K_MAX:
            continue
        X = scaler.transform(df[INPUT_FEATURE_COLUMNS].values.astype(np.float32))
        for t in range(context, len(df) - K_MAX):
            if max_windows and n_eval >= max_windows:
                break
            feats_ctx = torch.from_numpy(X[t - context:t]).unsqueeze(0).to(device)
            future_real = torch.from_numpy(X[t:t + K_MAX]).unsqueeze(0).to(device)
            z_imagined = model.imagine(feats_ctx, K_MAX)
            s_hat = model.decode(z_imagined)
            err = ((s_hat - future_real) ** 2).mean(dim=-1).squeeze(0)   # (K_MAX,)
            for k in range(K_MAX):
                sq_err[k].append(float(err[k]))
            n_eval += 1

    mse_per_k = [float(np.mean(sq_err[k])) for k in range(K_MAX)]
    return mse_per_k, n_eval, arch


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = {
        "RSSM, frozen (as trained)": os.path.join(CKPT_DIR, "rssm_phase1_best.pt"),
        "RSSM, encoder unfrozen (Phase-1 weights AFTER Phase-2 fine-tuning)":
            os.path.join(CKPT_DIR, "rssm_unfrozen_phase1.pt"),
        "Baseline Transformer, frozen": os.path.join(CKPT_DIR, "phase1_best.pt"),
    }
    print(f"{'config':60s}  " + "  ".join(f"k={k+1}" for k in range(K_MAX)) + "   n_eval")
    results = {}
    for name, path in configs.items():
        if not os.path.exists(path):
            print(f"{name:60s}  (checkpoint not found: {path})")
            continue
        mse, n_eval, arch = eval_dynamics(path, device)
        results[name] = mse
        print(f"{name:60s}  " + "  ".join(f"{v:.4f}" for v in mse) + f"   {n_eval}")

    print("\nInterpretation: this is RAW, scaled-feature-space open-loop reconstruction MSE --")
    print("the same quantity the multi-step training loss directly optimizes, reported standalone")
    print("rather than folded into a downstream classification/forecast head's own loss.")
    if "RSSM, frozen (as trained)" in results and \
       "RSSM, encoder unfrozen (Phase-1 weights AFTER Phase-2 fine-tuning)" in results:
        frozen = np.array(results["RSSM, frozen (as trained)"])
        unfrozen = np.array(results["RSSM, encoder unfrozen (Phase-1 weights AFTER Phase-2 fine-tuning)"])
        delta = (unfrozen - frozen) / frozen * 100
        print(f"\nunfrozen vs frozen, % change in raw reconstruction MSE per horizon: "
             f"{[f'{d:+.1f}%' for d in delta]}")
        print("(positive = unfreezing made raw dynamics reconstruction WORSE, "
             "negative = better, independent of any downstream head)")


if __name__ == "__main__":
    main()
