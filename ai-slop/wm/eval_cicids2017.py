"""Zero-shot generalization test: the world model + stage head are trained
entirely on CIC-IDS-2018 + CTU-13 (see train.py / train_heads.py) and never
see a single CIC-IDS-2017 window during training or scaling. This evaluates
that already-trained checkpoint against CIC-IDS-2017 as a genuinely held-out,
independently-captured dataset -- the actual test of "learned real dynamics"
vs. "memorized 2018-specific artifacts" that dataset.md's usage plan called
for.

The scaler is loaded from the Phase-1 checkpoint (fit only on 2018+CTU-13
train splits) and applied as-is -- refitting it on 2017 data would leak
2017's own statistics into what is supposed to be an unseen-distribution
test.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import WorldModel, WorldModelConfig  # noqa: E402
from heads import StageHead, N_STAGES  # noqa: E402
from train_heads import macro_f1  # noqa: E402
from data import RobustScaler  # noqa: E402
from features import INPUT_FEATURE_COLUMNS  # noqa: E402

CKPT_DIR = "/media/kavinder/hdd2/sih26153-processed/checkpoints"
CICIDS2017_DIR = "/media/kavinder/hdd2/sih26153-processed/cicids2017"


def load_everything(device):
    ckpt1 = torch.load(os.path.join(CKPT_DIR, "phase1_best.pt"), map_location=device)
    cfg = WorldModelConfig(**ckpt1["cfg"])
    model = WorldModel(cfg).to(device)
    model.load_state_dict(ckpt1["model"])
    model.eval()

    ckpt2 = torch.load(os.path.join(CKPT_DIR, "phase2_head.pt"), map_location=device)
    head = StageHead(cfg.d_latent).to(device)
    head.load_state_dict(ckpt2["head"])
    head.eval()

    scaler = RobustScaler()
    scaler.center = ckpt1["scaler_center"].cpu().numpy()
    scaler.scale = ckpt1["scaler_scale"].cpu().numpy()
    context = ckpt1["context"]
    return model, head, scaler, context


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, head, scaler, context = load_everything(device)

    preds_real, labels_real = [], []
    preds_imag, labels_imag = [], []
    per_day_summary = []

    for path in sorted(glob.glob(os.path.join(CICIDS2017_DIR, "*.features.csv"))):
        day = os.path.basename(path).replace(".features.csv", "")
        df = pd.read_csv(path)
        if len(df) <= context:
            continue
        X = scaler.transform(df[INPUT_FEATURE_COLUMNS].values.astype(np.float32))
        y = df["stage_idx"].values.astype(np.int64)
        feats = torch.from_numpy(X).unsqueeze(0).to(device)

        z = model.encode(feats)
        pred_real_day = head(z[:, :context]).argmax(-1).cpu().numpy().ravel()
        label_real_day = y[:context]

        horizon = len(df) - context
        z_imag = model.imagine(feats[:, :context], horizon)
        pred_imag_day = head(z_imag).argmax(-1).cpu().numpy().ravel()
        label_imag_day = y[context:]

        preds_real.append(pred_real_day); labels_real.append(label_real_day)
        preds_imag.append(pred_imag_day); labels_imag.append(label_imag_day)

        f1_day, _ = macro_f1(pred_imag_day, label_imag_day, N_STAGES)
        stages_present = sorted(set(df["stage"].unique()))
        per_day_summary.append((day, len(df), stages_present, f1_day))

    preds_real, labels_real = np.concatenate(preds_real), np.concatenate(labels_real)
    preds_imag, labels_imag = np.concatenate(preds_imag), np.concatenate(labels_imag)
    f1_real, pc_real = macro_f1(preds_real, labels_real, N_STAGES)
    f1_imag, pc_imag = macro_f1(preds_imag, labels_imag, N_STAGES)

    print("=== CIC-IDS-2017 zero-shot (never trained on) ===\n")
    print(f"macroF1_real={f1_real:.3f}   macroF1_imagined(rolled-forward)={f1_imag:.3f}\n")
    print("per-class (real-context positions):")
    for stage, m in pc_real.items():
        print(f"  {stage:20s} P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  n={m['support']}")
    print("\nper-class (imagined/rolled-forward positions):")
    for stage, m in pc_imag.items():
        print(f"  {stage:20s} P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  n={m['support']}")
    print("\nper-day summary (imagined-position macro-F1):")
    for day, n, stages, f1_day in per_day_summary:
        print(f"  {day:45s} n={n:4d}  stages={stages}  macroF1={f1_day:.3f}")


if __name__ == "__main__":
    main()
