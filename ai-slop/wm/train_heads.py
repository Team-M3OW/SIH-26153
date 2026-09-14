"""Phase 2: train the stage head on top of the frozen Phase-1 world model,
and fit the Pi transition matrix. Encoder + transition are frozen -- only
the head's parameters get gradients.
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
from data import build_labeled_datasets, load_days, temporal_split, RobustScaler  # noqa: E402
from heads import (StageHead, InfilHead, fit_transition_matrix,  # noqa: E402
                  rollout_transition_matrix, N_STAGES, INFIL_IDX)
from arch_loader import load_world_model  # noqa: E402

CKPT_DIR = "/media/kavinder/hdd2/sih26153-processed/checkpoints"
PHASE1_CKPT = os.path.join(CKPT_DIR, "phase1_best.pt")


def stage_loss(model, head, feats, labels, ctx, class_weight=None):
    """Cross-entropy over real context positions AND imagined horizon
    positions, concatenated into one batch -- not two weighted terms."""
    with torch.no_grad():
        z = model.encode(feats)                      # (B, ctx+H, d_latent)
    logits_real = head(z[:, :ctx])                    # real encoded states
    z_imagined = model.imagine(feats[:, :ctx], feats.shape[1] - ctx)  # (B, H, d_latent)
    logits_imag = head(z_imagined)                    # states the model made up

    logits = torch.cat([logits_real, logits_imag], dim=1)
    loss = F.cross_entropy(logits.reshape(-1, N_STAGES), labels.reshape(-1),
                           weight=class_weight)
    return loss, logits_real, logits_imag


def infil_loss(model, infil_head, feats, labels, ctx, pos_weight=None):
    """Binary cross-entropy, same real+imagined concatenation as stage_loss,
    but its own separate head, its own separate loss -- never combined with
    the stage classifier's loss."""
    with torch.no_grad():
        z = model.encode(feats)
    logits_real = infil_head(z[:, :ctx])
    z_imagined = model.imagine(feats[:, :ctx], feats.shape[1] - ctx)
    logits_imag = infil_head(z_imagined)

    logits = torch.cat([logits_real, logits_imag], dim=1)
    y = torch.isin(labels, INFIL_IDX.to(labels.device)).float()
    loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
    return loss


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


@torch.no_grad()
def evaluate_infil(model, infil_head, loader, device, ctx):
    model.eval(); infil_head.eval()
    p_real, y_real, p_imag, y_imag = [], [], [], []
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        y = torch.isin(labels, INFIL_IDX.to(device)).float()
        z = model.encode(feats)
        p_real.append(torch.sigmoid(infil_head(z[:, :ctx])).cpu().numpy().ravel())
        y_real.append(y[:, :ctx].cpu().numpy().ravel())

        z_imag = model.imagine(feats[:, :ctx], feats.shape[1] - ctx)
        p_imag.append(torch.sigmoid(infil_head(z_imag)).cpu().numpy().ravel())
        y_imag.append(y[:, ctx:].cpu().numpy().ravel())
    infil_head.train()
    p_real, y_real = np.concatenate(p_real), np.concatenate(y_real)
    p_imag, y_imag = np.concatenate(p_imag), np.concatenate(y_imag)
    return brier_score(p_real, y_real), brier_score(p_imag, y_imag)


def macro_f1(preds: np.ndarray, labels: np.ndarray, n_classes: int) -> tuple[float, dict]:
    """Unweighted mean of per-class F1 -- classes with zero support are
    excluded from the mean rather than counted as 0, so a stage this dataset
    never contains (Recon, Exfiltration) doesn't silently drag the score
    down for a reason that has nothing to do with model quality."""
    from attack_map import STAGES
    per_class = {}
    f1s = []
    for c in range(n_classes):
        support = (labels == c).sum()
        if support == 0:
            continue
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[STAGES[c]] = {"precision": prec, "recall": rec, "f1": f1, "support": int(support)}
        f1s.append(f1)
    return (sum(f1s) / len(f1s) if f1s else 0.0), per_class


@torch.no_grad()
def evaluate(model, head, loader, device, ctx, n_classes=N_STAGES):
    model.eval(); head.eval()
    preds_real, labels_real, preds_imag, labels_imag = [], [], [], []
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        z = model.encode(feats)
        preds_real.append(head(z[:, :ctx]).argmax(-1).cpu().numpy().ravel())
        labels_real.append(labels[:, :ctx].cpu().numpy().ravel())

        z_imag = model.imagine(feats[:, :ctx], feats.shape[1] - ctx)
        preds_imag.append(head(z_imag).argmax(-1).cpu().numpy().ravel())
        labels_imag.append(labels[:, ctx:].cpu().numpy().ravel())
    head.train()
    preds_real, labels_real = np.concatenate(preds_real), np.concatenate(labels_real)
    preds_imag, labels_imag = np.concatenate(preds_imag), np.concatenate(labels_imag)
    f1_real, per_class_real = macro_f1(preds_real, labels_real, n_classes)
    f1_imag, per_class_imag = macro_f1(preds_imag, labels_imag, n_classes)
    return f1_real, f1_imag, per_class_real, per_class_imag


def main(epochs=20, batch_size=128, lr=1e-3, phase1_ckpt=PHASE1_CKPT):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, ckpt, arch = load_world_model(phase1_ckpt, device)
    print(f"Phase-1 architecture: {arch}")
    for p in model.parameters():
        p.requires_grad_(False)

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
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    infil_head = InfilHead(cfg.d_latent).to(device)
    infil_opt = torch.optim.AdamW(infil_head.parameters(), lr=lr, weight_decay=1e-4)

    # Sqrt-inverse-frequency class weights from the train split -- plain
    # (unweighted) cross-entropy collapses to always predicting Benign;
    # raw inverse-frequency weighting (Impact ~25x Benign) overcorrects into
    # high recall but very low precision on every minority class. Sqrt
    # softens the correction, trading some of that recall back for
    # precision without returning all the way to the collapse.
    train_labels = np.concatenate([s["train"]["stage_idx"].values for s in splits.values()])
    counts = np.bincount(train_labels, minlength=N_STAGES).astype(np.float64)
    raw_weight = np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 0.0)
    class_weight = torch.tensor(np.sqrt(raw_weight), dtype=torch.float32, device=device)

    # sqrt-softened, same fix already applied to the stage head's
    # class_weight above (raw inverse-frequency overcorrected there too).
    infil_rate = np.isin(train_labels, INFIL_IDX.numpy()).mean()
    infil_raw_weight = (1 - infil_rate) / max(infil_rate, 1e-6)
    infil_pos_weight = torch.tensor(np.sqrt(infil_raw_weight), dtype=torch.float32, device=device)
    print(f"infil_rate={infil_rate:.3f}  raw_pos_weight={infil_raw_weight:.2f}  "
          f"sqrt_pos_weight={infil_pos_weight.item():.2f}")

    best_val, best_state = -1.0, None
    best_infil_val, best_infil_state = float("inf"), None
    for epoch in range(epochs):
        t0 = time.time()
        for feats, labels in loaders["train"]:
            feats, labels = feats.to(device), labels.to(device)
            loss, _, _ = stage_loss(model, head, feats, labels, context, class_weight)
            opt.zero_grad(); loss.backward(); opt.step()

            il = infil_loss(model, infil_head, feats, labels, context, infil_pos_weight)
            infil_opt.zero_grad(); il.backward(); infil_opt.step()

        f1_real, f1_imag, _, _ = evaluate(model, head, loaders["val"], device, context)
        brier_real, brier_imag = evaluate_infil(model, infil_head, loaders["val"], device, context)
        # min(), not f1_imag alone -- selecting on one metric let a
        # checkpoint through where Command & Control had collapsed to 0 on
        # real positions while imagined ticked up. min() refuses any
        # checkpoint where a class has silently collapsed on either side.
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
    for stage, m in pc_real.items():
        print(f"  {stage:20s} P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  n={m['support']}")
    print("\nper-class (imagined/rolled-forward positions):")
    for stage, m in pc_imag.items():
        print(f"  {stage:20s} P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  n={m['support']}")

    # Pi: counted from real labeled sequences in the train split, per day,
    # never crossing a day boundary.
    train_stage_seqs = [s["train"]["stage_idx"].values for s in splits.values()
                        if len(s["train"]) > 1]
    pi = fit_transition_matrix(train_stage_seqs)
    print("\nPi (rows=from, cols=to):")
    from attack_map import STAGES
    print("           " + "  ".join(f"{s[:10]:>10s}" for s in STAGES))
    for i, row in enumerate(pi):
        print(f"{STAGES[i]:10s} " + "  ".join(f"{v:10.3f}" for v in row))

    out_name = "phase2_head.pt" if arch == "transformer" else f"{arch}_phase2_head.pt"
    torch.save({"head": head.state_dict(), "infil_head": infil_head.state_dict(),
               "pi": torch.from_numpy(pi), "d_latent": cfg.d_latent},
              os.path.join(CKPT_DIR, out_name))
    print(f"\nsaved: {os.path.join(CKPT_DIR, out_name)}")
    return model, head, pi, scaler, splits, context, horizon


if __name__ == "__main__":
    main()
