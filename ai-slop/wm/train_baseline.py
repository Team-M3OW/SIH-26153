"""The mandated baseline: logistic regression on the identical flattened
context window and identical target as the world model, so any gap between
them is attributable to learned dynamics, not to different inputs or
different targets.

No temporal structure at all: the whole (context x n_features) window is
flattened into one vector per example. It is scored on exactly the target
`evaluate()`'s "imagined" metric in train_heads.py uses for the neural
model -- the stage label at each of the H windows after the context,
pooled the same way -- so macro-F1 is directly comparable between the two.

Same scaler as the world model (fit once, from the Phase-1 checkpoint,
never refit here) -- the baseline must see identical features, not just a
similar preprocessing recipe.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import (build_labeled_datasets, load_days, temporal_split,  # noqa: E402
                  RobustScaler)
from heads import N_STAGES  # noqa: E402
from train_heads import macro_f1, PHASE1_CKPT  # noqa: E402
from features import INPUT_FEATURE_COLUMNS  # noqa: E402

CICIDS2017_DIR = "/media/kavinder/hdd2/sih26153-processed/cicids2017"


def flatten_dataset(ds, context: int):
    """(B, context+H, F) sequences -> (N, context*F) flattened-context rows,
    each repeated once per horizon position, paired with that position's
    real label -- the same pooling `evaluate()` uses for the neural model's
    "imagined" macro-F1."""
    Xs, ys = [], []
    for i in range(len(ds)):
        seq, labels = ds[i]
        ctx_flat = seq[:context].reshape(-1).numpy()
        horizon_labels = labels[context:].numpy()
        for y in horizon_labels:
            Xs.append(ctx_flat)
            ys.append(y)
    return np.stack(Xs), np.array(ys)


def main():
    device = "cpu"
    ckpt1 = torch.load(PHASE1_CKPT, map_location=device)
    context, horizon = ckpt1["context"], ckpt1["horizon"]

    scaler = RobustScaler()
    scaler.center = ckpt1["scaler_center"].numpy()
    scaler.scale = ckpt1["scaler_scale"].numpy()

    days = load_days()
    splits = {name: temporal_split(df) for name, df in days.items()}
    datasets = build_labeled_datasets(context, horizon, scaler, splits)

    print("flattening sequences into (context * n_features) rows ...")
    X_train, y_train = flatten_dataset(datasets["train"], context)
    X_test, y_test = flatten_dataset(datasets["test"], context)
    print(f"train: {X_train.shape}   test: {X_test.shape}")

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)

    pred_test = clf.predict(X_test)
    f1_test, pc_test = macro_f1(pred_test, y_test, N_STAGES)
    print(f"\n=== logistic regression baseline: in-distribution TEST ===")
    print(f"macroF1 = {f1_test:.3f}  (world model's imagined macroF1 was 0.350)\n")
    for stage, m in pc_test.items():
        print(f"  {stage:20s} P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  n={m['support']}")

    # Zero-shot CIC-IDS-2017 -- same protocol as eval_cicids2017.py: one
    # flattened context window per day, used to predict every remaining
    # window in that day (the baseline has no evolving state to update, so
    # the same static vector stands in for all horizon positions -- this is
    # not a handicap we're imposing on it, it's the actual, fair test of
    # whether dynamics modeling beats a context snapshot with nothing else).
    import glob
    import pandas as pd
    Xs2017, ys2017 = [], []
    for path in sorted(glob.glob(os.path.join(CICIDS2017_DIR, "*.features.csv"))):
        df = pd.read_csv(path)
        if len(df) <= context:
            continue
        Xall = scaler.transform(df[INPUT_FEATURE_COLUMNS].values.astype(np.float32))
        y = df["stage_idx"].values.astype(np.int64)
        ctx_flat = Xall[:context].reshape(-1)
        for label in y[context:]:
            Xs2017.append(ctx_flat)
            ys2017.append(label)
    X_2017, y_2017 = np.stack(Xs2017), np.array(ys2017)
    pred_2017 = clf.predict(X_2017)
    f1_2017, pc_2017 = macro_f1(pred_2017, y_2017, N_STAGES)
    print(f"\n=== logistic regression baseline: CIC-IDS-2017 zero-shot ===")
    print(f"macroF1 = {f1_2017:.3f}  (world model's zero-shot macroF1 was 0.047)\n")
    for stage, m in pc_2017.items():
        print(f"  {stage:20s} P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  n={m['support']}")


if __name__ == "__main__":
    main()
