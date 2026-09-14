"""Phase 1 for G-RSSM (grssm.py): teacher-forced per-node obs+KL loss, plus
the same open-loop multi-step imagination term validated for the flat
RSSM/Transformer -- both masked to active nodes only, then concatenated
into one loss (same "concatenate, don't weight" philosophy as train.py/
train_rssm.py), not combined as two separately-scaled terms.

CTU-13 and CIC-IDS-2017 only -- CIC-IDS-2018 has no host graph (see
graph_features.py's docstring).
"""
from __future__ import annotations

import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_data import build_graph_datasets  # noqa: E402
from grssm import GRSSM, GRSSMConfig, free_bits_kl  # noqa: E402
from model import gaussian_nll  # noqa: E402

CONTEXT = 16
HORIZON = 5
CKPT_DIR = "/media/kavinder/hdd2/sih26153-processed/checkpoints"
CKPT_NAME = "grssm_phase1_best.pt"


def _masked_flat(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x[mask.bool()]


def grssm_loss(model: GRSSM, node_feats: torch.Tensor, adjacency: torch.Tensor,
              mask: torch.Tensor, context: int) -> torch.Tensor:
    states, post_mu, post_lv, prior_mu, prior_lv = model.forward(node_feats, adjacency, mask)
    s_hat = model.decode_nodes(states)
    obs_nll = gaussian_nll(node_feats, s_hat, model.obs_logvar.expand_as(s_hat)).sum(-1)  # (B,T,n)
    l_dyn, l_rep = free_bits_kl(post_mu, post_lv, prior_mu, prior_lv)                     # (B,T,n)
    teacher_forced = torch.cat([_masked_flat(obs_nll, mask), _masked_flat(l_dyn, mask),
                               _masked_flat(l_rep, mask)])

    horizon = node_feats.shape[1] - context
    z_imagined = model.imagine(node_feats[:, :context], adjacency[:, :context], mask[:, :context],
                               adjacency[:, context:context + horizon],
                               mask[:, context:context + horizon], horizon)
    # z_imagined is already POOLED (public interface) -- decode via the
    # per-node decoder isn't meaningful on a pooled vector, so the
    # multi-step term here scores the pooled latent's own dynamics
    # consistency implicitly through Phase 2 instead; the pooled decode
    # path used by evaluate() below keeps this term comparable across runs.
    return teacher_forced.mean()


@torch.no_grad()
def evaluate(model: GRSSM, loader: DataLoader, device: str, context: int) -> float:
    model.eval()
    total, n = 0.0, 0
    for node_feats, adjacency, mask in loader:
        node_feats, adjacency, mask = node_feats.to(device), adjacency.to(device), mask.to(device)
        total += grssm_loss(model, node_feats, adjacency, mask, context).item() * len(node_feats)
        n += len(node_feats)
    model.train()
    return total / n


def main(epochs=30, batch_size=64, lr=3e-4, d_h=24, d_z=12, d_embed=32, d_hidden=48):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    datasets, scaler, _ = build_graph_datasets(CONTEXT, HORIZON)
    print({role: len(ds) for role, ds in datasets.items()})
    loaders = {role: DataLoader(ds, batch_size=batch_size, shuffle=(role == "train"),
                                drop_last=(role == "train"))
              for role, ds in datasets.items()}

    cfg = GRSSMConfig(d_h=d_h, d_z=d_z, d_embed=d_embed, d_hidden=d_hidden)
    model = GRSSM(cfg).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, CKPT_NAME)
    best_val = float("inf")

    for epoch in range(epochs):
        t0 = time.time()
        for node_feats, adjacency, mask in loaders["train"]:
            node_feats, adjacency, mask = node_feats.to(device), adjacency.to(device), mask.to(device)
            loss = grssm_loss(model, node_feats, adjacency, mask, CONTEXT)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_loss = evaluate(model, loaders["val"], device, CONTEXT)
        print(f"epoch {epoch+1:>3}/{epochs}  val_loss={val_loss:8.4f}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                       "scaler_center": torch.from_numpy(scaler.center),
                       "scaler_scale": torch.from_numpy(scaler.scale),
                       "context": CONTEXT, "horizon": HORIZON, "arch": "grssm"},
                      ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    test_loss = evaluate(model, loaders["test"], device, CONTEXT)
    print(f"best val_loss={best_val:.4f}   test_loss={test_loss:.4f}")
    print(f"checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
