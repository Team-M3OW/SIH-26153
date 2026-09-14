"""The check flagged repeatedly all session and never run: CIC-IDS-2018 has
exactly two Infiltration episodes (Wednesday-28-02-2018 and
Thursday-01-03-2018, both mapped to Lateral Movement by attack_map.py).
Every LR-baseline number reported all session trained on windows pooled
across ALL days (including both Infiltration days) and tested on a later
TIME SLICE of the same days -- so the ~0.995 AUC LR has been getting on
cic_ids2018 could be genuine "infiltration looks like X" skill, or could be
memorizing something specific to the one or two episodes it was trained on
(a particular host's traffic shape, a specific timing pattern) that
doesn't generalize to a THIRD, unseen infiltration.

This is the strongest test available without a third real episode: train
one per-horizon LR purely on ONE full day, test purely on the OTHER full
day it has never seen -- true episode-level holdout, not just a later time
slice of days it's already seen parts of. Run in both directions.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_days, RobustScaler  # noqa: E402
from features import INPUT_FEATURE_COLUMNS  # noqa: E402
from attack_map import INFILTRATION_STAGES  # noqa: E402

CONTEXT = 16
K_MAX = 5
DAY_A = "Wednesday-28-02-2018"
DAY_B = "Thursday-01-03-2018"


def windows(df: pd.DataFrame, scaler: RobustScaler, context: int, k_max: int):
    X = scaler.transform(df[INPUT_FEATURE_COLUMNS].values.astype(np.float32))
    infil = df["stage"].isin(INFILTRATION_STAGES).values.astype(np.int64)
    Xs, Ys = [], []
    for t in range(context, len(df) - k_max):
        Xs.append(X[t - context:t].reshape(-1))
        Ys.append(infil[t: t + k_max])
    return np.stack(Xs), np.stack(Ys)


def run_direction(train_df, test_df, label):
    scaler = RobustScaler().fit(train_df[INPUT_FEATURE_COLUMNS].values.astype(np.float32))
    Xtr, Ytr = windows(train_df, scaler, CONTEXT, K_MAX)
    Xte, Yte = windows(test_df, scaler, CONTEXT, K_MAX)
    print(f"\n=== {label} === train_n={len(Xtr)}  test_n={len(Xte)}")
    print(f"{'k':>3}  {'train_pos_rate':>14}  {'test_pos_rate':>14}  {'brier':>8}  {'auc':>8}")
    for k in range(K_MAX):
        ytr_k, yte_k = Ytr[:, k], Yte[:, k]
        train_rate, test_rate = ytr_k.mean(), yte_k.mean()
        if len(np.unique(ytr_k)) < 2:
            print(f"{k+1:>3}  {train_rate:>14.3f}  {test_rate:>14.3f}  {'--no positive class in train--':>8}")
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, ytr_k)
        p = clf.predict_proba(Xte)[:, 1]
        brier = float(np.mean((p - yte_k) ** 2))
        if len(np.unique(yte_k)) < 2:
            print(f"{k+1:>3}  {train_rate:>14.3f}  {test_rate:>14.3f}  {brier:>8.4f}  {'--insufficient test class variety--'}")
            continue
        auc = roc_auc_score(yte_k, p)
        print(f"{k+1:>3}  {train_rate:>14.3f}  {test_rate:>14.3f}  {brier:>8.4f}  {auc:>8.4f}")


def main():
    days = load_days(["/media/kavinder/hdd2/sih26153-processed/cic_ids2018"])
    df_a, df_b = days[DAY_A], days[DAY_B]
    print(f"{DAY_A}: {len(df_a)} windows, infiltration frac={df_a['stage'].isin(INFILTRATION_STAGES).mean():.3f}")
    print(f"{DAY_B}: {len(df_b)} windows, infiltration frac={df_b['stage'].isin(INFILTRATION_STAGES).mean():.3f}")

    run_direction(df_a, df_b, f"train={DAY_A}  test={DAY_B}")
    run_direction(df_b, df_a, f"train={DAY_B}  test={DAY_A}")


if __name__ == "__main__":
    main()
