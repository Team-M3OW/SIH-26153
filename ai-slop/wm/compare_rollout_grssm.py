"""Same four-way infiltration-forecast comparison as compare_rollout.py
(neural/derived/pi/lr), run against G-RSSM instead -- see that file's
docstring for what each method means. Only two families here (ctu13,
cicids2017), since CIC-IDS-2018 has no host graph.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_data import load_graph_days, temporal_split, NodeFeatureScaler, build_labeled_graph_datasets  # noqa: E402
from grssm import GRSSM, GRSSMConfig  # noqa: E402
from heads import StageHead, InfilHead, rollout_transition_matrix, INFIL_IDX, N_STAGES  # noqa: E402
from attack_map import INFILTRATION_STAGES, STAGE_TO_IDX  # noqa: E402

CKPT_DIR = "/media/kavinder/hdd2/sih26153-processed/checkpoints"
K_MAX = 5
INFIL_IDX_NP = np.array([STAGE_TO_IDX[s] for s in INFILTRATION_STAGES])


def load_everything(device, phase1_ckpt=None, phase2_ckpt=None):
    phase1_ckpt = phase1_ckpt or os.path.join(CKPT_DIR, "grssm_phase1_best.pt")
    ckpt1 = torch.load(phase1_ckpt, map_location=device)
    cfg = GRSSMConfig(**ckpt1["cfg"])
    model = GRSSM(cfg).to(device)
    model.load_state_dict(ckpt1["model"])
    model.eval()

    phase2_ckpt = phase2_ckpt or os.path.join(CKPT_DIR, "grssm_phase2_head.pt")
    ckpt2 = torch.load(phase2_ckpt, map_location=device)
    head = StageHead(cfg.d_latent).to(device)
    head.load_state_dict(ckpt2["head"]); head.eval()
    infil_head = InfilHead(cfg.d_latent).to(device)
    infil_head.load_state_dict(ckpt2["infil_head"]); infil_head.eval()
    pi = ckpt2["pi"].cpu().numpy()

    scaler = NodeFeatureScaler()
    scaler.center = ckpt1["scaler_center"].cpu().numpy()
    scaler.scale = ckpt1["scaler_scale"].cpu().numpy()
    context = ckpt1["context"]
    return model, head, infil_head, pi, scaler, context


def train_lr_per_horizon(splits, scaler, context, k_max):
    Xs, ys = [], []
    for name, s in splits.items():
        d = s["train"]
        n = len(d["stage_idx"])
        nf = scaler.transform(d["node_feats"])
        infil = np.isin(d["stage_idx"], INFIL_IDX_NP).astype(np.int64)
        span = context + k_max
        for t in range(0, n - span + 1):
            Xs.append(nf[t:t + context].reshape(-1))
            ys.append(infil[t + context: t + span])
    if not Xs:
        return [None] * k_max
    X = np.stack(Xs)
    Y = np.stack(ys)
    models = []
    for k in range(k_max):
        y_k = Y[:, k]
        if len(np.unique(y_k)) < 2:
            models.append(None)
        else:
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(X, y_k)
            models.append(clf)
    return models


def forecast_lr(models, feats_ctx_flat, k_max):
    out = np.zeros(k_max)
    for k in range(k_max):
        clf = models[k]
        out[k] = clf.predict_proba(feats_ctx_flat.reshape(1, -1))[0, 1] if clf else 0.0
    return out


@torch.no_grad()
def forecast_one(model, head, infil_head, pi, node_feats_ctx, adjacency_ctx, mask_ctx,
                 adjacency_future, mask_future, k_max, device, n_traj=64):
    z_ctx = model.encode(node_feats_ctx, adjacency_ctx, mask_ctx)
    gamma_t = torch.softmax(head(z_ctx[:, -1]), dim=-1).squeeze(0).cpu().numpy()

    out = model.rollout(node_feats_ctx, adjacency_ctx, mask_ctx, adjacency_future, mask_future,
                        k=k_max, n_traj=n_traj)
    lat = out["latents"].squeeze(0)

    stage_probs = torch.softmax(head(lat), dim=-1).mean(dim=1)
    infil_derived = stage_probs.index_select(-1, INFIL_IDX.to(device)).sum(-1).cpu().numpy()
    infil_neural = torch.sigmoid(infil_head(lat)).mean(dim=1).cpu().numpy()
    infil_pi = np.array([
        rollout_transition_matrix(pi, gamma_t, k)[INFIL_IDX.numpy()].sum()
        for k in range(1, k_max + 1)
    ])
    return infil_neural, infil_pi, infil_derived


def main(k_max=K_MAX, n_traj=64, max_windows=None, phase1_ckpt=None, phase2_ckpt=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, head, infil_head, pi, scaler, context = load_everything(device, phase1_ckpt, phase2_ckpt)

    raw = load_graph_days()
    splits = {name: temporal_split(d) for name, d in raw.items()}

    print("training per-horizon logistic regression baseline ...")
    lr_models = train_lr_per_horizon(splits, scaler, context, k_max)

    methods = ["neural", "derived", "pi", "lr"]
    families = ["all", "cicids2017", "ctu13"]
    sq = {fam: {k: {m: [] for m in methods} for k in range(k_max)} for fam in families}
    p = {fam: {k: {m: [] for m in methods} for k in range(k_max)} for fam in families}
    y_true = {fam: {k: [] for k in range(k_max)} for fam in families}
    n_eval = 0

    for name, s in splits.items():
        d = s["test"]
        n = len(d["stage_idx"])
        if n < context + k_max:
            continue
        fam = "ctu13" if name.startswith("scenario") else "cicids2017"
        nf = scaler.transform(d["node_feats"])
        infil_true = np.isin(d["stage_idx"], INFIL_IDX_NP).astype(np.float32)

        for t in range(context, n - k_max):
            if max_windows and n_eval >= max_windows:
                break
            node_feats_ctx = torch.from_numpy(nf[t - context:t]).unsqueeze(0).to(device)
            adjacency_ctx = torch.from_numpy(d["adjacency"][t - context:t]).unsqueeze(0).to(device)
            mask_ctx = torch.from_numpy(d["mask"][t - context:t]).unsqueeze(0).to(device)
            adjacency_future = torch.from_numpy(d["adjacency"][t:t + k_max]).unsqueeze(0).to(device)
            mask_future = torch.from_numpy(d["mask"][t:t + k_max]).unsqueeze(0).to(device)

            infil_neural, infil_pi, infil_derived = forecast_one(
                model, head, infil_head, pi, node_feats_ctx, adjacency_ctx, mask_ctx,
                adjacency_future, mask_future, k_max, device, n_traj)
            infil_lr = forecast_lr(lr_models, nf[t - context:t], k_max)
            truth = infil_true[t: t + k_max]
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
                return roc_auc_score(y, np.array(p[fam][k][m]))
            print(f"{k+1:>3}  " + "  ".join(f"{auc(m):>9.4f}" for m in methods))
        print()


if __name__ == "__main__":
    main()
