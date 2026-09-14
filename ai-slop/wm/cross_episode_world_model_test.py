"""The same true episode-level holdout just run for LR (cross_episode_lr_test.py),
now for the world model: RSSM trained end-to-end (Phase 1 self-supervised +
Phase 2 unfrozen classification/infiltration heads) with ONE of the two
CIC-IDS-2018 Infiltration episodes held out ENTIRELY -- not just its later
temporal-split test slice, but removed from training altogether -- then
evaluated purely on that untouched day. Both directions, matching the LR
test's structure exactly so the two are directly comparable.

This is more expensive than the LR test (full Phase 1 + Phase 2 retraining
per direction) but answers the same question for the architecture that
matters: does the world model's forecasting skill transfer to a genuinely
unseen infiltration episode, and how does that compare to LR's true
cross-episode AUC (0.68-0.82, see cross_episode_lr_test.py's results)?
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import (PROCESSED_DIRS, load_days, temporal_split, RobustScaler,  # noqa: E402
                  build_labeled_datasets, SequenceDataset)
from rssm import RSSM, RSSMConfig  # noqa: E402
from train_rssm import rssm_loss, evaluate as evaluate_rssm  # noqa: E402
from train_heads_unfrozen import stage_loss_unfrozen, infil_loss_unfrozen  # noqa: E402
from heads import StageHead, InfilHead, N_STAGES, INFIL_IDX  # noqa: E402
from features import INPUT_FEATURE_COLUMNS  # noqa: E402

CONTEXT = 16
HORIZON = 5
DAY_A = "Wednesday-28-02-2018"
DAY_B = "Thursday-01-03-2018"


def build_holdout_splits(holdout_day: str) -> dict:
    days = load_days(PROCESSED_DIRS)
    splits = {}
    for name, df in days.items():
        if name == holdout_day:
            empty = df.iloc[0:0]
            splits[name] = {"train": empty, "val": empty, "test": df}
        else:
            splits[name] = temporal_split(df)
    return splits


def train_phase1(splits: dict, device: str, epochs=30, batch_size=128, lr=3e-4):
    train_feats = np.concatenate(
        [s["train"][INPUT_FEATURE_COLUMNS].values.astype(np.float32)
         for s in splits.values() if len(s["train"])])
    scaler = RobustScaler().fit(train_feats)

    def arrays_for(role):
        kept = [s[role] for s in splits.values() if len(s[role]) >= CONTEXT + HORIZON]
        return [scaler.transform(df[INPUT_FEATURE_COLUMNS].values.astype(np.float32)) for df in kept]

    train_ds = SequenceDataset(arrays_for("train"), CONTEXT, HORIZON)
    val_ds = SequenceDataset(arrays_for("val"), CONTEXT, HORIZON)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    cfg = RSSMConfig(n_features=len(scaler.center))
    model = RSSM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        for batch in train_loader:
            batch = batch.to(device)
            loss = rssm_loss(model, batch, CONTEXT)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_loss = evaluate_rssm(model, val_loader, device, CONTEXT)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"  Phase 1 done, best_val_loss={best_val:.4f}", flush=True)
    return model, cfg, scaler


def train_phase2_unfrozen(model, cfg, scaler, splits, device, epochs=20, batch_size=128, lr=1e-3):
    for p in model.parameters():
        p.requires_grad_(True)
    datasets = build_labeled_datasets(CONTEXT, HORIZON, scaler, splits)
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False)

    head = StageHead(cfg.d_latent).to(device)
    infil_head = InfilHead(cfg.d_latent).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    infil_opt = torch.optim.AdamW(list(model.parameters()) + list(infil_head.parameters()), lr=lr, weight_decay=1e-4)

    train_stage = np.concatenate([s["train"]["stage_idx"].values for s in splits.values() if len(s["train"])])
    counts = np.bincount(train_stage, minlength=N_STAGES).astype(np.float64)
    raw_weight = np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 0.0)
    class_weight = torch.tensor(np.sqrt(raw_weight), dtype=torch.float32, device=device)
    infil_rate = np.isin(train_stage, INFIL_IDX.numpy()).mean()
    infil_pos_weight = torch.tensor(np.sqrt((1 - infil_rate) / max(infil_rate, 1e-6)),
                                    dtype=torch.float32, device=device)

    best_val, best_model_state, best_head_state, best_infil_state = float("inf"), None, None, None
    for epoch in range(epochs):
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            loss, _, _ = stage_loss_unfrozen(model, head, feats, labels, CONTEXT, class_weight)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            il = infil_loss_unfrozen(model, infil_head, feats, labels, CONTEXT, infil_pos_weight)
            infil_opt.zero_grad(); il.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(infil_head.parameters()), 1.0)
            infil_opt.step()

        with torch.no_grad():
            model.eval(); infil_head.eval()
            total, n = 0.0, 0
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                y = torch.isin(labels, INFIL_IDX.to(device)).float()
                z = model.encode(feats)
                p_real = torch.sigmoid(infil_head(z[:, :CONTEXT]))
                total += ((p_real - y[:, :CONTEXT]) ** 2).sum().item()
                n += y[:, :CONTEXT].numel()
            val_brier = total / max(n, 1)
            model.train(); infil_head.train()
        if val_brier < best_val:
            best_val = val_brier
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_head_state = {k: v.clone() for k, v in head.state_dict().items()}
            best_infil_state = {k: v.clone() for k, v in infil_head.state_dict().items()}

    model.load_state_dict(best_model_state)
    head.load_state_dict(best_head_state)
    infil_head.load_state_dict(best_infil_state)
    print(f"  Phase 2 done, best_val_infilBrier={best_val:.4f}", flush=True)
    return model, head, infil_head


@torch.no_grad()
def evaluate_holdout(model, infil_head, scaler, holdout_df, device, k_max=5):
    model.eval(); infil_head.eval()
    X = scaler.transform(holdout_df[INPUT_FEATURE_COLUMNS].values.astype(np.float32))
    from attack_map import INFILTRATION_STAGES
    infil_true = holdout_df["stage"].isin(INFILTRATION_STAGES).values.astype(np.float32)

    p_by_k = {k: [] for k in range(k_max)}
    y_by_k = {k: [] for k in range(k_max)}
    for t in range(CONTEXT, len(holdout_df) - k_max):
        feats_ctx = torch.from_numpy(X[t - CONTEXT:t]).unsqueeze(0).to(device)
        out = model.rollout(feats_ctx, k=k_max, n_traj=64)
        lat = out["latents"].squeeze(0)
        probs = torch.sigmoid(infil_head(lat)).mean(dim=1).cpu().numpy()
        truth = infil_true[t + 1: t + 1 + k_max]
        for k in range(k_max):
            p_by_k[k].append(probs[k])
            y_by_k[k].append(truth[k])

    print(f"{'k':>3}  {'brier':>8}  {'auc':>8}  n_eval={len(p_by_k[0])}")
    for k in range(k_max):
        p, y = np.array(p_by_k[k]), np.array(y_by_k[k])
        brier = float(np.mean((p - y) ** 2))
        if len(np.unique(y)) < 2:
            print(f"{k+1:>3}  {brier:>8.4f}  {'--insufficient--':>8}")
            continue
        auc = roc_auc_score(y, p)
        print(f"{k+1:>3}  {brier:>8.4f}  {auc:>8.4f}")


def run_direction(holdout_day: str, device: str):
    print(f"\n{'='*20} holdout = {holdout_day} {'='*20}")
    splits = build_holdout_splits(holdout_day)
    t0 = time.time()
    model, cfg, scaler = train_phase1(splits, device)
    model, head, infil_head = train_phase2_unfrozen(model, cfg, scaler, splits, device)
    print(f"  training took {time.time()-t0:.1f}s")

    holdout_df = load_days(PROCESSED_DIRS)[holdout_day]
    evaluate_holdout(model, infil_head, scaler, holdout_df, device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_direction(DAY_A, device)   # train on everything except Wed-28, test purely on Wed-28
    run_direction(DAY_B, device)   # train on everything except Thu-01, test purely on Thu-01


if __name__ == "__main__":
    main()
