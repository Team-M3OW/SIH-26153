"""Phase 2 for G-RSSM: same StageHead/InfilHead (unchanged, they only ever
see the pooled flat d_latent vector) as train_heads.py, but the loss
functions call GRSSM's own encode/imagine signature (needs adjacency+mask
alongside the node features) instead of the flat model's single-tensor
call. Encoder frozen, same as the flat pipeline's default -- see session
history on why freezing suppresses rare-class signal; an unfrozen variant
could be added the same way train_heads_unfrozen.py did, not built here to
keep this experiment scoped to "does the graph structure itself help."
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_data import build_labeled_graph_datasets, load_graph_days, temporal_split, NodeFeatureScaler  # noqa: E402
from grssm import GRSSM, GRSSMConfig  # noqa: E402
from heads import StageHead, InfilHead, fit_transition_matrix, N_STAGES, INFIL_IDX  # noqa: E402
from train_heads import macro_f1  # noqa: E402

CKPT_DIR = "/media/kavinder/hdd2/sih26153-processed/checkpoints"
PHASE1_CKPT = os.path.join(CKPT_DIR, "grssm_phase1_best.pt")


def stage_loss(model, head, node_feats, adjacency, mask, labels, ctx, class_weight=None):
    with torch.no_grad():
        z = model.encode(node_feats, adjacency, mask)
    logits_real = head(z[:, :ctx])
    horizon = node_feats.shape[1] - ctx
    z_imagined = model.imagine(node_feats[:, :ctx], adjacency[:, :ctx], mask[:, :ctx],
                               adjacency[:, ctx:ctx + horizon], mask[:, ctx:ctx + horizon], horizon)
    logits_imag = head(z_imagined)
    logits = torch.cat([logits_real, logits_imag], dim=1)
    loss = F.cross_entropy(logits.reshape(-1, N_STAGES), labels.reshape(-1), weight=class_weight)
    return loss, logits_real, logits_imag


def infil_loss(model, infil_head, node_feats, adjacency, mask, labels, ctx, pos_weight=None):
    with torch.no_grad():
        z = model.encode(node_feats, adjacency, mask)
    logits_real = infil_head(z[:, :ctx])
    horizon = node_feats.shape[1] - ctx
    z_imagined = model.imagine(node_feats[:, :ctx], adjacency[:, :ctx], mask[:, :ctx],
                               adjacency[:, ctx:ctx + horizon], mask[:, ctx:ctx + horizon], horizon)
    logits_imag = infil_head(z_imagined)
    logits = torch.cat([logits_real, logits_imag], dim=1)
    y = torch.isin(labels, INFIL_IDX.to(labels.device)).float()
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)


def brier_score(probs, labels):
    return float(np.mean((probs - labels) ** 2))


@torch.no_grad()
def evaluate_infil(model, infil_head, loader, device, ctx):
    model.eval(); infil_head.eval()
    p_real, y_real, p_imag, y_imag = [], [], [], []
    for node_feats, adjacency, mask, stage, infil in loader:
        node_feats, adjacency, mask = node_feats.to(device), adjacency.to(device), mask.to(device)
        stage = stage.to(device)
        y = torch.isin(stage, INFIL_IDX.to(device)).float()
        z = model.encode(node_feats, adjacency, mask)
        p_real.append(torch.sigmoid(infil_head(z[:, :ctx])).cpu().numpy().ravel())
        y_real.append(y[:, :ctx].cpu().numpy().ravel())
        horizon = node_feats.shape[1] - ctx
        z_imag = model.imagine(node_feats[:, :ctx], adjacency[:, :ctx], mask[:, :ctx],
                               adjacency[:, ctx:ctx + horizon], mask[:, ctx:ctx + horizon], horizon)
        p_imag.append(torch.sigmoid(infil_head(z_imag)).cpu().numpy().ravel())
        y_imag.append(y[:, ctx:].cpu().numpy().ravel())
    infil_head.train()
    p_real, y_real = np.concatenate(p_real), np.concatenate(y_real)
    p_imag, y_imag = np.concatenate(p_imag), np.concatenate(y_imag)
    return brier_score(p_real, y_real), brier_score(p_imag, y_imag)


@torch.no_grad()
def evaluate(model, head, loader, device, ctx, n_classes=N_STAGES):
    model.eval(); head.eval()
    preds_real, labels_real, preds_imag, labels_imag = [], [], [], []
    for node_feats, adjacency, mask, stage, infil in loader:
        node_feats, adjacency, mask = node_feats.to(device), adjacency.to(device), mask.to(device)
        stage = stage.to(device)
        z = model.encode(node_feats, adjacency, mask)
        logits_real = head(z[:, :ctx])
        preds_real.append(logits_real.argmax(-1).cpu().numpy().ravel())
        labels_real.append(stage[:, :ctx].cpu().numpy().ravel())
        horizon = node_feats.shape[1] - ctx
        z_imag = model.imagine(node_feats[:, :ctx], adjacency[:, :ctx], mask[:, :ctx],
                               adjacency[:, ctx:ctx + horizon], mask[:, ctx:ctx + horizon], horizon)
        logits_imag = head(z_imag)
        preds_imag.append(logits_imag.argmax(-1).cpu().numpy().ravel())
        labels_imag.append(stage[:, ctx:].cpu().numpy().ravel())
    head.train()
    preds_real, labels_real = np.concatenate(preds_real), np.concatenate(labels_real)
    preds_imag, labels_imag = np.concatenate(preds_imag), np.concatenate(labels_imag)
    f1_real, pc_real = macro_f1(preds_real, labels_real, n_classes)
    f1_imag, pc_imag = macro_f1(preds_imag, labels_imag, n_classes)
    return f1_real, f1_imag, pc_real, pc_imag


def main(epochs=20, batch_size=64, lr=1e-3, phase1_ckpt=PHASE1_CKPT):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(phase1_ckpt, map_location=device)
    cfg = GRSSMConfig(**ckpt["cfg"])
    model = GRSSM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    for p in model.parameters():
        p.requires_grad_(False)
    context, horizon = ckpt["context"], ckpt["horizon"]

    scaler = NodeFeatureScaler()
    scaler.center = ckpt["scaler_center"].cpu().numpy()
    scaler.scale = ckpt["scaler_scale"].cpu().numpy()

    raw = load_graph_days()
    splits = {name: temporal_split(d) for name, d in raw.items()}
    datasets = build_labeled_graph_datasets(context, horizon, scaler, splits)
    print({role: len(ds) for role, ds in datasets.items()})
    loaders = {role: DataLoader(ds, batch_size=batch_size, shuffle=(role == "train"),
                                drop_last=(role == "train"))
              for role, ds in datasets.items()}

    head = StageHead(cfg.d_latent).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    infil_head = InfilHead(cfg.d_latent).to(device)
    infil_opt = torch.optim.AdamW(infil_head.parameters(), lr=lr, weight_decay=1e-4)

    train_labels = np.concatenate([s["train"]["stage_idx"] for s in splits.values()])
    counts = np.bincount(train_labels, minlength=N_STAGES).astype(np.float64)
    raw_weight = np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 0.0)
    class_weight = torch.tensor(np.sqrt(raw_weight), dtype=torch.float32, device=device)

    infil_rate = np.isin(train_labels, INFIL_IDX.numpy()).mean()
    infil_raw_weight = (1 - infil_rate) / max(infil_rate, 1e-6)
    infil_pos_weight = torch.tensor(np.sqrt(infil_raw_weight), dtype=torch.float32, device=device)
    print(f"infil_rate={infil_rate:.3f}  sqrt_pos_weight={infil_pos_weight.item():.2f}")

    best_val, best_state = -1.0, None
    best_infil_val, best_infil_state = float("inf"), None
    for epoch in range(epochs):
        t0 = time.time()
        for node_feats, adjacency, mask, stage, infil in loaders["train"]:
            node_feats, adjacency, mask = node_feats.to(device), adjacency.to(device), mask.to(device)
            stage = stage.to(device)
            loss, _, _ = stage_loss(model, head, node_feats, adjacency, mask, stage, context, class_weight)
            opt.zero_grad(); loss.backward(); opt.step()

            il = infil_loss(model, infil_head, node_feats, adjacency, mask, stage, context, infil_pos_weight)
            infil_opt.zero_grad(); il.backward(); infil_opt.step()

        f1_real, f1_imag, _, _ = evaluate(model, head, loaders["val"], device, context)
        brier_real, brier_imag = evaluate_infil(model, infil_head, loaders["val"], device, context)
        score = min(f1_real, f1_imag)
        print(f"epoch {epoch+1:>3}/{epochs}  val_macroF1_real={f1_real:.3f}  "
              f"val_macroF1_imagined={f1_imag:.3f}  val_infilBrier_real={brier_real:.4f}  "
              f"val_infilBrier_imagined={brier_imag:.4f}  ({time.time()-t0:.1f}s)", flush=True)
        if score > best_val:
            best_val = score
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        if brier_imag < best_infil_val:
            best_infil_val = brier_imag
            best_infil_state = {k: v.clone() for k, v in infil_head.state_dict().items()}

    head.load_state_dict(best_state)
    infil_head.load_state_dict(best_infil_state)
    ib_real, ib_imag = evaluate_infil(model, infil_head, loaders["test"], device, context)
    print(f"\nTEST  infilBrier_real={ib_real:.4f}  infilBrier_imagined={ib_imag:.4f}")
    f1_real, f1_imag, pc_real, pc_imag = evaluate(model, head, loaders["test"], device, context)
    print(f"\nTEST  macroF1_real={f1_real:.3f}  macroF1_imagined={f1_imag:.3f}")
    print("\nper-class (real-context positions):")
    for stage_name, m in pc_real.items():
        print(f"  {stage_name:20s} P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  n={m['support']}")
    print("\nper-class (imagined/rolled-forward positions):")
    for stage_name, m in pc_imag.items():
        print(f"  {stage_name:20s} P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  n={m['support']}")

    train_stage_seqs = [s["train"]["stage_idx"] for s in splits.values() if len(s["train"]["stage_idx"]) > 1]
    pi = fit_transition_matrix(train_stage_seqs)

    out_path = os.path.join(CKPT_DIR, "grssm_phase2_head.pt")
    torch.save({"head": head.state_dict(), "infil_head": infil_head.state_dict(),
               "pi": torch.from_numpy(pi), "d_latent": cfg.d_latent}, out_path)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
