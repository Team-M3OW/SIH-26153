"""5-seed replication of the best/most-defensible configuration: RSSM,
encoder unfrozen, NO DANN (DANN's headline result didn't replicate across
seeds -- see session history; this checks whether the plain unfrozen
config itself is stable, which was never directly tested with more than
one seed).

Skips retraining the LR baseline per seed (it doesn't depend on the world
model checkpoint and is expensive) -- this script is purely about the
world model's own seed-to-seed variance. Seed 1 reuses the checkpoint
already on disk (rssm_unfrozen_phase1.pt / rssm_unfrozen_phase2_head.pt);
seeds 2-5 are trained fresh here.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_rssm  # noqa: E402
import train_heads_unfrozen as th_unfrozen  # noqa: E402
from train_heads import evaluate, evaluate_infil  # noqa: E402
from data import load_days, temporal_split, RobustScaler, build_labeled_datasets  # noqa: E402
from heads import StageHead, InfilHead, rollout_transition_matrix, INFIL_IDX, N_STAGES  # noqa: E402
from attack_map import INFILTRATION_STAGES, STAGE_TO_IDX  # noqa: E402
from features import INPUT_FEATURE_COLUMNS  # noqa: E402
from arch_loader import load_world_model  # noqa: E402
from torch.utils.data import DataLoader

CKPT_DIR = "/media/kavinder/hdd2/sih26153-processed/checkpoints"
CONTEXT, HORIZON, K_MAX = 16, 5, 5
INFIL_IDX_NP = np.array([STAGE_TO_IDX[s] for s in INFILTRATION_STAGES])


@torch.no_grad()
def pooled_infil_metrics(model, head, infil_head, pi, scaler, context, device, n_traj=64, max_windows=None):
    """Pooled Brier/AUC of the 'neural' infiltration forecast, all domains
    pooled -- same computation as compare_rollout.py's forecast_one +
    aggregation loop, minus LR and the per-family breakdown."""
    from sklearn.metrics import roc_auc_score
    days = load_days()
    splits = {name: temporal_split(df) for name, df in days.items()}
    sq = {k: [] for k in range(K_MAX)}
    p = {k: [] for k in range(K_MAX)}
    y_true = {k: [] for k in range(K_MAX)}
    n_eval = 0
    for name, s in splits.items():
        df = s["test"]
        if len(df) < context + K_MAX:
            continue
        X = scaler.transform(df[INPUT_FEATURE_COLUMNS].values.astype(np.float32))
        infil_true = df["infiltration"].values.astype(np.float32)
        for t in range(context, len(df) - K_MAX):
            if max_windows and n_eval >= max_windows:
                break
            feats_ctx = torch.from_numpy(X[t - context:t]).unsqueeze(0).to(device)
            z_ctx = model.encode(feats_ctx)
            gamma_t = torch.softmax(head(z_ctx[:, -1]), dim=-1).squeeze(0).cpu().numpy()
            out = model.rollout(feats_ctx, k=K_MAX, n_traj=n_traj)
            lat = out["latents"].squeeze(0)
            infil_neural = torch.sigmoid(infil_head(lat)).mean(dim=1).cpu().numpy()
            truth = infil_true[t + 1: t + 1 + K_MAX]
            for k in range(K_MAX):
                sq[k].append((infil_neural[k] - truth[k]) ** 2)
                p[k].append(infil_neural[k])
                y_true[k].append(truth[k])
            n_eval += 1

    briers = [float(np.mean(sq[k])) for k in range(K_MAX)]
    aucs = [float(roc_auc_score(y_true[k], p[k])) if len(set(y_true[k])) > 1 else float("nan")
           for k in range(K_MAX)]
    return briers, aucs, n_eval


def eval_seed_full(phase1_ckpt, phase2_ckpt, device):
    model, cfg, ckpt1, arch = load_world_model(phase1_ckpt, device)
    ckpt2 = torch.load(phase2_ckpt, map_location=device)
    head = StageHead(cfg.d_latent).to(device); head.load_state_dict(ckpt2["head"]); head.eval()
    infil_head = InfilHead(cfg.d_latent).to(device); infil_head.load_state_dict(ckpt2["infil_head"]); infil_head.eval()
    pi = ckpt2["pi"].cpu().numpy()
    scaler = RobustScaler()
    scaler.center = ckpt1["scaler_center"].cpu().numpy()
    scaler.scale = ckpt1["scaler_scale"].cpu().numpy()
    context = ckpt1["context"]

    briers, aucs, n_eval = pooled_infil_metrics(model, head, infil_head, pi, scaler, context, device)

    days = load_days()
    splits = {name: temporal_split(df) for name, df in days.items()}
    labeled = build_labeled_datasets(context, HORIZON, scaler, splits)
    loader = DataLoader(labeled["test"], batch_size=128, shuffle=False)
    f1_real, f1_imag, _, _ = evaluate(model, head, loader, device, context)
    brier_real, brier_imag = evaluate_infil(model, infil_head, loader, device, context)

    return {
        "pooled_brier": briers, "pooled_auc": aucs, "n_eval": n_eval,
        "macroF1_real": f1_real, "macroF1_imag": f1_imag,
        "infilBrier_real": brier_real, "infilBrier_imag": brier_imag,
    }


def run_new_seed(seed_tag: str, device: str):
    t0 = time.time()
    train_rssm.CKPT_NAME = f"rssm_phase1_{seed_tag}_best.pt"
    train_rssm.main()
    phase1_ckpt = os.path.join(CKPT_DIR, train_rssm.CKPT_NAME)

    th_unfrozen.main(phase1_ckpt=phase1_ckpt, tag=f"unfrozen_{seed_tag}")
    # train_heads_unfrozen.main names outputs f"{arch}_{tag}_..." where arch
    # comes from the LOADED checkpoint ("rssm" for a plain train_rssm.py output).
    phase1_out = os.path.join(CKPT_DIR, f"rssm_unfrozen_{seed_tag}_phase1.pt")
    phase2_out = os.path.join(CKPT_DIR, f"rssm_unfrozen_{seed_tag}_phase2_head.pt")

    metrics = eval_seed_full(phase1_out, phase2_out, device)
    print(f"  [{seed_tag}] done in {time.time()-t0:.0f}s")
    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}

    print("=== seed1 (existing checkpoint) ===")
    results["seed1"] = eval_seed_full(
        os.path.join(CKPT_DIR, "rssm_unfrozen_phase1.pt"),
        os.path.join(CKPT_DIR, "rssm_unfrozen_phase2_head.pt"), device)
    print(results["seed1"])

    for i in range(2, 6):
        tag = f"ms{i}"
        print(f"=== seed{i} ({tag}) ===")
        results[f"seed{i}"] = run_new_seed(tag, device)
        print(results[f"seed{i}"])

    print("\n\n========== SUMMARY ACROSS 5 SEEDS ==========")
    macroF1_real = [results[s]["macroF1_real"] for s in results]
    macroF1_imag = [results[s]["macroF1_imag"] for s in results]
    pooled_auc_k1 = [results[s]["pooled_auc"][0] for s in results]
    pooled_auc_k5 = [results[s]["pooled_auc"][4] for s in results]
    pooled_brier_k1 = [results[s]["pooled_brier"][0] for s in results]
    pooled_brier_k5 = [results[s]["pooled_brier"][4] for s in results]

    def stat(name, vals):
        print(f"{name:22s}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
              f"min={np.min(vals):.4f}  max={np.max(vals):.4f}  values={[round(v,4) for v in vals]}")

    stat("macroF1_real", macroF1_real)
    stat("macroF1_imagined", macroF1_imag)
    stat("pooled AUC k=1", pooled_auc_k1)
    stat("pooled AUC k=5", pooled_auc_k5)
    stat("pooled Brier k=1", pooled_brier_k1)
    stat("pooled Brier k=5", pooled_brier_k5)

    import json
    with open(os.path.join(os.path.dirname(__file__), "multiseed_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nfull results -> multiseed_results.json")


if __name__ == "__main__":
    main()
