"""Compare four ways of producing a K-step infiltration-probability forecast:

  neural   -- the dedicated InfilHead, applied to 64 sampled open-loop
              trajectories, averaged. Directly supervised on the binary
              target, same as `lr` below.
  derived  -- the ORIGINAL mechanism: sum the stage classifier's softmax
              mass over {Lateral Movement, C2, Exfiltration}. Never
              directly supervised on this sum -- kept only so the two
              neural mechanisms can be compared head-to-head.
  pi       -- closed-form (Pi^T)^k applied to the current stage posterior --
              the cheap, interpretable one.
  lr       -- the mandated baseline, given every fair chance: one SEPARATE
              binary logistic regression per horizon k, each trained
              specifically on the k-step-ahead infiltration label.

Platt-scaling calibration was tried here and made things WORSE (see
docs/architecture.md) -- the validation and test splits have different
attack base rates (a known issue from the class-imbalance investigation),
so a calibrator fit on val transfers the wrong correction to test. Removed
rather than left in as a misleading "fix".

All four are scored identically: Brier score and AUC of the predicted
infiltration probability against the real ground-truth infiltration label
at t+k, over the exact same context windows in the test split. Nothing here
is trained on the test split; this is pure comparison, not fitting.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_days, temporal_split, RobustScaler, build_labeled_datasets  # noqa: E402
from heads import StageHead, InfilHead, rollout_transition_matrix, INFIL_IDX, N_STAGES  # noqa: E402
from features import INPUT_FEATURE_COLUMNS  # noqa: E402
from attack_map import INFILTRATION_STAGES, STAGE_TO_IDX  # noqa: E402
from arch_loader import load_world_model  # noqa: E402

CKPT_DIR = "/media/kavinder/hdd2/sih26153-processed/checkpoints"
K_MAX = 5
INFIL_IDX_NP = np.array([STAGE_TO_IDX[s] for s in INFILTRATION_STAGES])


def load_everything(device, phase1_ckpt=None, phase2_ckpt=None):
    phase1_ckpt = phase1_ckpt or os.path.join(CKPT_DIR, "phase1_best.pt")
    model, cfg, ckpt1, arch = load_world_model(phase1_ckpt, device)
    print(f"loaded architecture: {arch}")

    phase2_ckpt = phase2_ckpt or os.path.join(
        CKPT_DIR, "phase2_head.pt" if arch == "transformer" else f"{arch}_phase2_head.pt")
    ckpt2 = torch.load(phase2_ckpt, map_location=device)
    head = StageHead(cfg.d_latent).to(device)
    head.load_state_dict(ckpt2["head"])
    head.eval()
    infil_head = InfilHead(cfg.d_latent).to(device)
    infil_head.load_state_dict(ckpt2["infil_head"])
    infil_head.eval()
    pi = ckpt2["pi"].cpu().numpy()

    scaler = RobustScaler()
    scaler.center = ckpt1["scaler_center"].cpu().numpy()
    scaler.scale = ckpt1["scaler_scale"].cpu().numpy()
    context = ckpt1["context"]
    return model, head, infil_head, pi, scaler, context


def train_lr_per_horizon(scaler, context, horizon, k_max):
    """One binary logistic regression per k, each trained specifically on
    the k-step-ahead infiltration label -- not one model averaged over all
    horizons. Same flattened-context input as train_baseline.py."""
    days = load_days()
    splits = {name: temporal_split(df) for name, df in days.items()}
    datasets = build_labeled_datasets(context, horizon, scaler, splits)
    train_ds = datasets["train"]

    ctx_flat = np.stack([train_ds[i][0][:context].reshape(-1).numpy()
                         for i in range(len(train_ds))])
    stage_labels = np.stack([train_ds[i][1].numpy() for i in range(len(train_ds))])
    infil_labels = np.isin(stage_labels, INFIL_IDX_NP).astype(np.int64)

    models = []
    for k in range(1, k_max + 1):
        y_k = infil_labels[:, context + k - 1]
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        if len(np.unique(y_k)) < 2:
            clf = None   # degenerate horizon (e.g. no positives) -- skip honestly
        else:
            clf.fit(ctx_flat, y_k)
        models.append(clf)
    return models


def forecast_lr(models, feats_ctx_flat: np.ndarray, k_max: int) -> np.ndarray:
    out = np.zeros(k_max)
    for k in range(k_max):
        clf = models[k]
        out[k] = clf.predict_proba(feats_ctx_flat.reshape(1, -1))[0, 1] if clf else 0.0
    return out


@torch.no_grad()
def forecast_one(model, head, infil_head, pi, feats_ctx: torch.Tensor, k_max: int,
                 device, n_traj=64):
    """feats_ctx: (1, context, n_features), already scaled.

    Returns (infil_neural, infil_pi, infil_derived) -- see module docstring."""
    z_ctx = model.encode(feats_ctx)
    gamma_t = torch.softmax(head(z_ctx[:, -1]), dim=-1).squeeze(0).cpu().numpy()  # (N_STAGES,)

    out = model.rollout(feats_ctx, k=k_max, n_traj=n_traj)   # latents: (1, K, n_traj, d_latent)
    lat = out["latents"].squeeze(0)                          # (K, n_traj, d_latent)

    stage_probs = torch.softmax(head(lat), dim=-1).mean(dim=1)   # (K, N_STAGES)
    infil_derived = stage_probs.index_select(-1, INFIL_IDX.to(device)).sum(-1).cpu().numpy()

    infil_neural = torch.sigmoid(infil_head(lat)).mean(dim=1).cpu().numpy()   # (K,)

    infil_pi = np.array([
        rollout_transition_matrix(pi, gamma_t, k)[INFIL_IDX.numpy()].sum()
        for k in range(1, k_max + 1)
    ])
    return infil_neural, infil_pi, infil_derived


def main(k_max=K_MAX, n_traj=64, max_windows=None, phase1_ckpt=None, phase2_ckpt=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, head, infil_head, pi, scaler, context = load_everything(
        device, phase1_ckpt, phase2_ckpt)

    print("training per-horizon logistic regression baseline ...")
    lr_models = train_lr_per_horizon(scaler, context, k_max, k_max)

    days = load_days()
    splits = {name: temporal_split(df) for name, df in days.items()}

    methods = ["neural", "derived", "pi", "lr"]
    families = ["all", "cic_ids2018", "cic_ids2017", "ctu13"]
    sq = {fam: {k: {m: [] for m in methods} for k in range(k_max)} for fam in families}
    p = {fam: {k: {m: [] for m in methods} for k in range(k_max)} for fam in families}
    y_true = {fam: {k: [] for k in range(k_max)} for fam in families}
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
            fam = "cic_ids2017"   # day-of-week-named files, no year in the name
        X = scaler.transform(df[INPUT_FEATURE_COLUMNS].values.astype(np.float32))
        infil_true = df["infiltration"].values.astype(np.float32)

        for t in range(context, len(df) - k_max):
            if max_windows and n_eval >= max_windows:
                break
            feats_ctx = torch.from_numpy(X[t - context:t]).unsqueeze(0).to(device)
            infil_neural, infil_pi, infil_derived = forecast_one(
                model, head, infil_head, pi, feats_ctx, k_max, device, n_traj)
            infil_lr = forecast_lr(lr_models, X[t - context:t], k_max)
            truth = infil_true[t + 1: t + 1 + k_max]
            for k in range(k_max):
                preds = {"neural": infil_neural[k], "derived": infil_derived[k],
                        "pi": infil_pi[k], "lr": infil_lr[k]}
                for name_, val in preds.items():
                    for f in ("all", fam):
                        sq[f][k][name_].append((val - truth[k]) ** 2)
                        p[f][k][name_].append(val)
                for f in ("all", fam):
                    y_true[f][k].append(truth[k])
            n_eval += 1

    print(f"\nevaluated {n_eval} context windows across {len(splits)} days' test splits\n")

    for fam in families:
        print(f"=== {fam} ===")
        print("Brier score (lower is better):")
        print(f"{'k':>3}  " + "  ".join(f"{m:>9}" for m in methods) + f"  {'winner':>9}")
        for k in range(k_max):
            if not y_true[fam][k]:
                continue
            briers = {m: np.mean(sq[fam][k][m]) for m in methods}
            winner = min(briers, key=briers.get)
            print(f"{k+1:>3}  " + "  ".join(f"{briers[m]:>9.4f}" for m in methods) + f"  {winner:>9}")

        print("AUC (ranking quality only, ignores calibration):")
        print(f"{'k':>3}  " + "  ".join(f"{m:>9}" for m in methods))
        for k in range(k_max):
            y = np.array(y_true[fam][k])
            if len(y) == 0 or len(np.unique(y)) < 2:
                print(f"{k+1:>3}  (insufficient class variety in this slice)")
                continue
            def auc(m, fam=fam, k=k, y=y):
                pr = np.array(p[fam][k][m])
                return roc_auc_score(y, pr)
            print(f"{k+1:>3}  " + "  ".join(f"{auc(m):>9.4f}" for m in methods))
        print()

    # One concrete, qualitative example: the real Infiltration onset in
    # Wednesday-28-02-2018, if it's in this day's test split; otherwise
    # skip -- most of that day's onset falls in its own train split.
    name = "Wednesday-28-02-2018"
    if name in splits:
        df = splits[name]["test"]
        onset_positions = np.where(df["infiltration"].values == 1)[0]
        if len(onset_positions) and onset_positions[0] >= context:
            t = int(onset_positions[0])
            X = scaler.transform(df[INPUT_FEATURE_COLUMNS].values.astype(np.float32))
            feats_ctx = torch.from_numpy(X[t - context:t]).unsqueeze(0).to(device)
            infil_neural, infil_pi, infil_derived = forecast_one(
                model, head, infil_head, pi, feats_ctx, k_max, device, n_traj)
            infil_lr = forecast_lr(lr_models, X[t - context:t], k_max)
            truth = df["infiltration"].values[t + 1: t + 1 + k_max]
            print(f"\nqualitative example: {name} test split, context ending at window {t}")
            print(f"{'k':>3}  {'neural':>8}  {'derived':>8}  {'pi':>8}  {'lr':>8}  {'truth':>6}")
            for k in range(k_max):
                print(f"{k+1:>3}  {infil_neural[k]:>8.3f}  {infil_derived[k]:>8.3f}  "
                      f"{infil_pi[k]:>8.3f}  {infil_lr[k]:>8.3f}  {truth[k]:>6.0f}")


if __name__ == "__main__":
    main()
