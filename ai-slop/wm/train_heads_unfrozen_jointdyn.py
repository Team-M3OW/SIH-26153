"""Fixes the trade-off measured in eval_dynamics_quality.py: plain
unfreezing (train_heads_unfrozen.py) trains the now-unfrozen encoder
ONLY against the downstream classification/infiltration losses, with
nothing anchoring it back to the dynamics objective -- so it's free to
drift arbitrarily far from good reconstruction, and measurably does
(raw MSE +350-408%). This adds the SAME multi-step dynamics loss used in
Phase 1 (train_rssm.rssm_loss) back into the SAME optimizer step,
concatenated with the classification loss (this project's established
"concatenate, don't weight" philosophy, same pattern as one-step+
multi-step and stage+infil elsewhere) -- so the encoder has to satisfy
both objectives simultaneously, not sequentially.

Added to the STAGE optimizer step only (not duplicated into the infil
step too), matching how domain-adversarial training was structured
earlier in this project (added once per batch, not into every loss term).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_labeled_datasets, load_days, temporal_split, RobustScaler  # noqa: E402
from heads import StageHead, InfilHead, fit_transition_matrix, N_STAGES, INFIL_IDX  # noqa: E402
from arch_loader import load_world_model  # noqa: E402
from train_heads import evaluate, evaluate_infil  # noqa: E402
from train_heads_unfrozen import stage_loss_unfrozen, infil_loss_unfrozen  # noqa: E402
from train_rssm import rssm_loss  # noqa: E402

CKPT_DIR = "/media/kavinder/hdd2/sih26153-processed/checkpoints"


def main(epochs=20, batch_size=128, lr=1e-3, phase1_ckpt=None, tag="unfrozen_jointdyn"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, ckpt, arch = load_world_model(phase1_ckpt, device)
    print(f"Phase-1 architecture: {arch}  (encoder UNFROZEN, joint dynamics+classification loss this run)")
    for p in model.parameters():
        p.requires_grad_(True)

    scaler = RobustScaler()
    scaler.center = ckpt["scaler_center"].cpu().numpy()
    scaler.scale = ckpt["scaler_scale"].cpu().numpy()
    context, horizon = ckpt["context"], ckpt["horizon"]

    days = load_days()
    splits = {name: temporal_split(df) for name, df in days.items()}
    datasets = build_labeled_datasets(context, horizon, scaler, splits)
    print({role: len(ds) for role, ds in datasets.items()})
    loaders = {role: DataLoader(ds, batch_size=batch_size, shuffle=(role == "train"),
                                drop_last=(role == "train"))
              for role, ds in datasets.items()}

    head = StageHead(cfg.d_latent).to(device)
    infil_head = InfilHead(cfg.d_latent).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    infil_opt = torch.optim.AdamW(list(model.parameters()) + list(infil_head.parameters()), lr=lr, weight_decay=1e-4)

    train_stage = np.concatenate([s["train"]["stage_idx"].values for s in splits.values()])
    counts = np.bincount(train_stage, minlength=N_STAGES).astype(np.float64)
    raw_weight = np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 0.0)
    class_weight = torch.tensor(np.sqrt(raw_weight), dtype=torch.float32, device=device)
    infil_rate = np.isin(train_stage, INFIL_IDX.numpy()).mean()
    infil_pos_weight = torch.tensor(np.sqrt((1 - infil_rate) / max(infil_rate, 1e-6)),
                                    dtype=torch.float32, device=device)
    print(f"infil_rate={infil_rate:.3f}  sqrt_pos_weight={infil_pos_weight.item():.2f}")

    best_val, best_model_state, best_head_state = -1.0, None, None
    best_val_infil, best_infil_state = float("inf"), None
    for epoch in range(epochs):
        t0 = time.time()
        for feats, labels in loaders["train"]:
            feats, labels = feats.to(device), labels.to(device)
            loss_stage, _, _ = stage_loss_unfrozen(model, head, feats, labels, context, class_weight)
            loss_dyn = rssm_loss(model, feats, context)
            loss = loss_stage + loss_dyn
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
            opt.step()

            il = infil_loss_unfrozen(model, infil_head, feats, labels, context, infil_pos_weight)
            infil_opt.zero_grad(); il.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(infil_head.parameters()), 1.0)
            infil_opt.step()

        f1_real, f1_imag, _, _ = evaluate(model, head, loaders["val"], device, context)
        brier_real, brier_imag = evaluate_infil(model, infil_head, loaders["val"], device, context)
        score = min(f1_real, f1_imag)
        print(f"epoch {epoch+1:>3}/{epochs}  val_macroF1_real={f1_real:.3f}  "
              f"val_macroF1_imagined={f1_imag:.3f}  val_infilBrier_real={brier_real:.4f}  "
              f"val_infilBrier_imagined={brier_imag:.4f}  ({time.time()-t0:.1f}s)", flush=True)
        if score > best_val:
            best_val = score
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_head_state = {k: v.clone() for k, v in head.state_dict().items()}
        if brier_imag < best_val_infil:
            best_val_infil = brier_imag
            best_infil_state = {k: v.clone() for k, v in infil_head.state_dict().items()}

    model.load_state_dict(best_model_state)
    head.load_state_dict(best_head_state)
    infil_head.load_state_dict(best_infil_state)

    ib_real, ib_imag = evaluate_infil(model, infil_head, loaders["test"], device, context)
    print(f"\nTEST  infilBrier_real={ib_real:.4f}  infilBrier_imagined={ib_imag:.4f}")
    f1_real, f1_imag, pc_real, pc_imag = evaluate(model, head, loaders["test"], device, context)
    print(f"\nTEST  macroF1_real={f1_real:.3f}  macroF1_imagined={f1_imag:.3f}")
    print("\nper-class (real-context positions):")
    for stage, m in pc_real.items():
        print(f"  {stage:20s} P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  n={m['support']}")
    print("\nper-class (imagined/rolled-forward positions):")
    for stage, m in pc_imag.items():
        print(f"  {stage:20s} P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  n={m['support']}")

    train_stage_seqs = [s["train"]["stage_idx"].values for s in splits.values() if len(s["train"]) > 1]
    pi = fit_transition_matrix(train_stage_seqs)

    model_ckpt_path = os.path.join(CKPT_DIR, f"{arch}_{tag}_phase1.pt")
    head_ckpt_path = os.path.join(CKPT_DIR, f"{arch}_{tag}_phase2_head.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
               "scaler_center": ckpt["scaler_center"], "scaler_scale": ckpt["scaler_scale"],
               "context": context, "horizon": horizon, "arch": f"{arch}_{tag}"},
              model_ckpt_path)
    torch.save({"head": head.state_dict(), "infil_head": infil_head.state_dict(),
               "pi": torch.from_numpy(pi), "d_latent": cfg.d_latent},
              head_ckpt_path)
    print(f"\nsaved: {model_ckpt_path}")
    print(f"saved: {head_ckpt_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--phase1_ckpt", required=True)
    args = p.parse_args()
    main(phase1_ckpt=args.phase1_ckpt)
