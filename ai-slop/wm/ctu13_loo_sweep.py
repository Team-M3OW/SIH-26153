"""Leave-one-scenario-out cross-episode sweep over CTU-13 -- giving the
cross-episode generalization claim genuine statistical power (10
independent held-out botnet scenarios, vs. the 2 CIC-2018 Infiltration
days in cross_episode_world_model_test.py). Reuses that script's
train-on-everything-except-holdout / evaluate-on-holdout machinery
directly, just looping over CTU-13 scenarios instead of the two CIC-2018
days.

3 of CTU-13's 13 scenarios (46, 48, 52) have fewer than 40 windows total
-- too few to serve as a standalone held-out test set (need context+k_max
= 21 windows just to form one evaluable context window) -- excluded as
holdout targets, though they remain in the training pool for every OTHER
scenario's run.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cross_episode_world_model_test import (  # noqa: E402
    build_holdout_splits, train_phase1, train_phase2_unfrozen, CONTEXT,
)
from data import load_days, PROCESSED_DIRS  # noqa: E402
from attack_map import INFILTRATION_STAGES  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

K_MAX = 5
MIN_HOLDOUT_WINDOWS = 40


@torch.no_grad()
def evaluate_holdout_full(model, infil_head, scaler, holdout_df, device, k_max=K_MAX):
    """Same computation as cross_episode_world_model_test.evaluate_holdout,
    but returns the numbers instead of only printing them."""
    from features import INPUT_FEATURE_COLUMNS
    model.eval(); infil_head.eval()
    X = scaler.transform(holdout_df[INPUT_FEATURE_COLUMNS].values.astype(np.float32))
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

    briers, aucs = [], []
    for k in range(k_max):
        p, y = np.array(p_by_k[k]), np.array(y_by_k[k])
        briers.append(float(np.mean((p - y) ** 2)))
        aucs.append(float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"))
    return briers, aucs, len(p_by_k[0])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctu13_days = load_days(["/media/kavinder/hdd2/sih26153-processed/ctu13"])
    holdout_candidates = sorted([name for name, df in ctu13_days.items()
                                if len(df) >= MIN_HOLDOUT_WINDOWS])
    print(f"holdout candidates ({len(holdout_candidates)} of {len(ctu13_days)} scenarios, "
         f">= {MIN_HOLDOUT_WINDOWS} windows): {holdout_candidates}\n", flush=True)

    results = {}
    for scenario in holdout_candidates:
        print(f"\n{'='*20} holdout = {scenario} {'='*20}", flush=True)
        t0 = time.time()
        splits = build_holdout_splits(scenario)
        model, cfg, scaler = train_phase1(splits, device)
        model, head, infil_head = train_phase2_unfrozen(model, cfg, scaler, splits, device)
        print(f"  training took {time.time()-t0:.0f}s", flush=True)

        holdout_df = ctu13_days[scenario]
        briers, aucs, n_eval = evaluate_holdout_full(model, infil_head, scaler, holdout_df, device)
        results[scenario] = {"brier": briers, "auc": aucs, "n_eval": n_eval}
        print(f"  n_eval={n_eval}  brier={[round(b,3) for b in briers]}  auc={[round(a,3) for a in aucs]}",
             flush=True)

        import json
        out_path = os.path.join(os.path.dirname(__file__), "ctu13_loo_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print("\n\n========== CTU-13 LEAVE-ONE-SCENARIO-OUT SUMMARY ==========")
    auc_k1 = [results[s]["auc"][0] for s in results if not np.isnan(results[s]["auc"][0])]
    auc_k5 = [results[s]["auc"][4] for s in results if not np.isnan(results[s]["auc"][4])]
    brier_k1 = [results[s]["brier"][0] for s in results]
    brier_k5 = [results[s]["brier"][4] for s in results]

    def stat(name, vals):
        print(f"{name:12s}  n={len(vals)}  mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  "
             f"min={np.min(vals):.3f}  max={np.max(vals):.3f}")

    stat("AUC k=1", auc_k1)
    stat("AUC k=5", auc_k5)
    stat("Brier k=1", brier_k1)
    stat("Brier k=5", brier_k5)

    import json
    out_path = os.path.join(os.path.dirname(__file__), "ctu13_loo_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nfull per-scenario results -> {out_path}")


if __name__ == "__main__":
    main()
