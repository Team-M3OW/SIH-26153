"""Per-stage forecasting breakdown -- unlike compare_rollout.py, which
collapses the stage distribution into one binary "infiltration" number
(sum over {Lateral Movement, C2, Exfiltration}), this reports Brier/AUC for
EACH individual stage's probability at each horizon k. Only `derived`
(stage-softmax over the rolled-out latent) and `pi` (Markov-chain rollout)
have a native per-stage probability -- `neural`/InfilHead is a dedicated
BINARY head with no per-stage decomposition, so it's not included here.

Reconnaissance and Initial Access are still reported if computable, but
expect them to hit the "insufficient class variety" guard often -- they
have only 8-22 and 34-424 test windows respectively across the whole
pooled test set, too few for a trustworthy per-horizon AUC (see session
history: this is a known, real data-scarcity limitation, not a bug here).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_days, temporal_split, RobustScaler  # noqa: E402
from heads import StageHead, rollout_transition_matrix, N_STAGES  # noqa: E402
from features import INPUT_FEATURE_COLUMNS  # noqa: E402
from attack_map import STAGES  # noqa: E402
from arch_loader import load_world_model  # noqa: E402

CKPT_DIR = "/media/kavinder/hdd2/sih26153-processed/checkpoints"
K_MAX = 5


def load_everything(device, phase1_ckpt, phase2_ckpt):
    model, cfg, ckpt1, arch = load_world_model(phase1_ckpt, device)
    print(f"loaded architecture: {arch}")
    ckpt2 = torch.load(phase2_ckpt, map_location=device)
    head = StageHead(cfg.d_latent).to(device)
    head.load_state_dict(ckpt2["head"]); head.eval()
    pi = ckpt2["pi"].cpu().numpy()

    scaler = RobustScaler()
    scaler.center = ckpt1["scaler_center"].cpu().numpy()
    scaler.scale = ckpt1["scaler_scale"].cpu().numpy()
    context = ckpt1["context"]
    return model, head, pi, scaler, context


@torch.no_grad()
def forecast_stage_probs(model, head, pi, feats_ctx, k_max, device, n_traj=64):
    """Returns (derived (K, N_STAGES), pi_probs (K, N_STAGES))."""
    z_ctx = model.encode(feats_ctx)
    gamma_t = torch.softmax(head(z_ctx[:, -1]), dim=-1).squeeze(0).cpu().numpy()

    out = model.rollout(feats_ctx, k=k_max, n_traj=n_traj)
    lat = out["latents"].squeeze(0)
    derived = torch.softmax(head(lat), dim=-1).mean(dim=1).cpu().numpy()   # (K, N_STAGES)

    pi_probs = np.stack([rollout_transition_matrix(pi, gamma_t, k) for k in range(1, k_max + 1)])
    return derived, pi_probs


def main(k_max=K_MAX, n_traj=64, max_windows=None, phase1_ckpt=None, phase2_ckpt=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    phase1_ckpt = phase1_ckpt or os.path.join(CKPT_DIR, "rssm_unfrozen_phase1.pt")
    phase2_ckpt = phase2_ckpt or os.path.join(CKPT_DIR, "rssm_unfrozen_phase2_head.pt")
    model, head, pi, scaler, context = load_everything(device, phase1_ckpt, phase2_ckpt)

    days = load_days()
    splits = {name: temporal_split(df) for name, df in days.items()}

    methods = ["derived", "pi"]
    families = ["all", "cic_ids2018", "cic_ids2017", "ctu13"]
    sq = {fam: {k: {m: {s: [] for s in STAGES} for m in methods} for k in range(k_max)} for fam in families}
    p = {fam: {k: {m: {s: [] for s in STAGES} for m in methods} for k in range(k_max)} for fam in families}
    y_true = {fam: {k: {s: [] for s in STAGES} for k in range(k_max)} for fam in families}
    n_eval = 0

    for name, s in splits.items():
        df = s["test"]
        if len(df) < context + k_max:
            continue
        if name.startswith("scenario"):
            fam = "ctu13"
        elif "2018" in name:
            fam = "cic_ids2018"
        else:
            fam = "cic_ids2017"
        X = scaler.transform(df[INPUT_FEATURE_COLUMNS].values.astype(np.float32))
        stage_true = df["stage"].values

        for t in range(context, len(df) - k_max):
            if max_windows and n_eval >= max_windows:
                break
            feats_ctx = torch.from_numpy(X[t - context:t]).unsqueeze(0).to(device)
            derived, pi_probs = forecast_stage_probs(model, head, pi, feats_ctx, k_max, device, n_traj)
            truth = stage_true[t + 1: t + 1 + k_max]
            for k in range(k_max):
                for si, stage in enumerate(STAGES):
                    y = 1.0 if truth[k] == stage else 0.0
                    for m, probs in (("derived", derived), ("pi", pi_probs)):
                        val = float(probs[k, si])
                        for f in ("all", fam):
                            sq[f][k][m][stage].append((val - y) ** 2)
                            p[f][k][m][stage].append(val)
                    for f in ("all", fam):
                        y_true[f][k][stage].append(y)
            n_eval += 1

    print(f"\nevaluated {n_eval} context windows across {len(splits)} days' test splits\n")

    for fam in families:
        print(f"=== {fam} ===")
        for stage in STAGES:
            print(f"--- {stage} ---")
            print(f"{'k':>3}  {'derived_brier':>13}  {'pi_brier':>10}  {'derived_auc':>12}  {'pi_auc':>8}  n_pos")
            for k in range(k_max):
                y = np.array(y_true[fam][k][stage])
                if len(y) == 0:
                    continue
                n_pos = int(y.sum())
                db = np.mean(sq[fam][k]["derived"][stage])
                pb = np.mean(sq[fam][k]["pi"][stage])
                if len(np.unique(y)) < 2:
                    print(f"{k+1:>3}  {db:>13.4f}  {pb:>10.4f}  {'--insufficient--':>12}  {'--':>8}  {n_pos}")
                    continue
                da = roc_auc_score(y, np.array(p[fam][k]["derived"][stage]))
                pa = roc_auc_score(y, np.array(p[fam][k]["pi"][stage]))
                print(f"{k+1:>3}  {db:>13.4f}  {pb:>10.4f}  {da:>12.4f}  {pa:>8.4f}  {n_pos}")
        print()


if __name__ == "__main__":
    main()
