"""Phase 2 variant: encoder UNFROZEN (see train_heads_unfrozen.py) AND
domain-adversarial (see train_rssm_domain.py's domain_loss/grl_lambda),
combined -- the direct fix for what unfreezing alone cost: RSSM's
cic_ids2017 domain-generalization AUC fell from 0.87-0.89 (frozen) to
0.62-0.65 (unfrozen) because fine-tuning on the classification loss pulled
the representation toward whatever the label-heavy domains (cic_ids2018,
ctu13) need. Adding a domain classifier trained adversarially against the
SAME unfrozen encoder, in the SAME fine-tuning step, pushes back against
exactly that pull while it's happening, rather than fixing it after the
fact.

Everything else identical to train_heads_unfrozen.py: same stage/infil
losses (still gradient-unblocked into the encoder), same lr/epochs/class
weighting. The only addition is the domain loss + GRL ramp, added to the
stage optimizer step only (matching train_rssm_domain.py's choice to add
domain_loss once per batch, not duplicated across every loss term).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_labeled_datasets, build_labeled_datasets_with_domain, load_days, temporal_split, RobustScaler, DOMAIN_NAMES  # noqa: E402
from heads import StageHead, InfilHead, fit_transition_matrix, N_STAGES, INFIL_IDX  # noqa: E402
from arch_loader import load_world_model  # noqa: E402
from model import DomainHead  # noqa: E402
from train import grl_lambda  # noqa: E402
from train_heads import evaluate, evaluate_infil  # noqa: E402
from train_heads_unfrozen import stage_loss_unfrozen, infil_loss_unfrozen  # noqa: E402

CKPT_DIR = "/media/kavinder/hdd2/sih26153-processed/checkpoints"


def domain_loss(model, domain_head, feats, domain_ids, lambd):
    z = model.encode(feats)
    logits = domain_head(z, lambd)
    targets = domain_ids.unsqueeze(1).expand(-1, logits.shape[1])
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def main(epochs=20, batch_size=128, lr=1e-3, phase1_ckpt=None, tag="unfrozen_dann"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, ckpt, arch = load_world_model(phase1_ckpt, device)
    print(f"Phase-1 architecture: {arch}  (encoder UNFROZEN + domain-adversarial this run)")
    for p in model.parameters():
        p.requires_grad_(True)

    scaler = RobustScaler()
    scaler.center = ckpt["scaler_center"].cpu().numpy()
    scaler.scale = ckpt["scaler_scale"].cpu().numpy()
    context, horizon = ckpt["context"], ckpt["horizon"]

    days = load_days()
    splits = {name: temporal_split(df) for name, df in days.items()}
    # Train needs domain_ids (for the domain loss); val/test evaluation
    # (evaluate/evaluate_infil, imported unchanged from train_heads.py)
    # only ever unpacks (feats, labels) -- so those use the domain-free
    # dataset, built from the exact same scaler/splits.
    train_ds = build_labeled_datasets_with_domain(context, horizon, scaler, splits)["train"]
    eval_ds = build_labeled_datasets(context, horizon, scaler, splits)
    print({"train": len(train_ds), "val": len(eval_ds["val"]), "test": len(eval_ds["test"])})
    loaders = {"train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
              "val": DataLoader(eval_ds["val"], batch_size=batch_size, shuffle=False),
              "test": DataLoader(eval_ds["test"], batch_size=batch_size, shuffle=False)}

    head = StageHead(cfg.d_latent).to(device)
    infil_head = InfilHead(cfg.d_latent).to(device)
    domain_head = DomainHead(cfg.d_latent, len(DOMAIN_NAMES)).to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()) + list(domain_head.parameters()),
        lr=lr, weight_decay=1e-4)
    infil_opt = torch.optim.AdamW(
        list(model.parameters()) + list(infil_head.parameters()), lr=lr, weight_decay=1e-4)

    train_labels = np.concatenate([s["train"]["stage_idx"].values for s in splits.values()])
    counts = np.bincount(train_labels, minlength=N_STAGES).astype(np.float64)
    raw_weight = np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 0.0)
    class_weight = torch.tensor(np.sqrt(raw_weight), dtype=torch.float32, device=device)

    infil_rate = np.isin(train_labels, INFIL_IDX.numpy()).mean()
    infil_raw_weight = (1 - infil_rate) / max(infil_rate, 1e-6)
    infil_pos_weight = torch.tensor(np.sqrt(infil_raw_weight), dtype=torch.float32, device=device)
    print(f"infil_rate={infil_rate:.3f}  sqrt_pos_weight={infil_pos_weight.item():.2f}")

    best_val, best_model_state, best_head_state = -1.0, None, None
    best_infil_val, best_infil_state = float("inf"), None
    n_batches = len(loaders["train"])
    for epoch in range(epochs):
        t0 = time.time()
        for bi, (feats, labels, domain_ids) in enumerate(loaders["train"]):
            feats, labels, domain_ids = feats.to(device), labels.to(device), domain_ids.to(device)
            progress = (epoch * n_batches + bi) / max(epochs * n_batches - 1, 1)
            lambd = grl_lambda(progress)

            loss, _, _ = stage_loss_unfrozen(model, head, feats, labels, context, class_weight)
            loss = loss + domain_loss(model, domain_head, feats[:, :context], domain_ids, lambd)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()) + list(domain_head.parameters()), 1.0)
            opt.step()

            il = infil_loss_unfrozen(model, infil_head, feats, labels, context, infil_pos_weight)
            infil_opt.zero_grad(); il.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(infil_head.parameters()), 1.0)
            infil_opt.step()

        f1_real, f1_imag, _, _ = evaluate(model, head, loaders["val"], device, context)
        brier_real, brier_imag = evaluate_infil(model, infil_head, loaders["val"], device, context)
        score = min(f1_real, f1_imag)
        print(f"epoch {epoch+1:>3}/{epochs}  val_macroF1_real={f1_real:.3f}  "
              f"val_macroF1_imagined={f1_imag:.3f}  val_infilBrier_real={brier_real:.4f}  "
              f"val_infilBrier_imagined={brier_imag:.4f}  lambda={lambd:.3f}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        if score > best_val:
            best_val = score
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_head_state = {k: v.clone() for k, v in head.state_dict().items()}
        if brier_imag < best_infil_val:
            best_infil_val = brier_imag
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

    train_stage_seqs = [s["train"]["stage_idx"].values for s in splits.values()
                        if len(s["train"]) > 1]
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
