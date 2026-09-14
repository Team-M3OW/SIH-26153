"""Phase 1 for RSSM + graph encoder ("G-RSSM"-style combination -- no paper
by that exact name was found; the closest verified precedent is OCRSSM,
which feeds a GNN's object-slot embeddings into an RSSM-style recurrent+
stochastic transition. Here the graph structure is over FEATURES, not
objects, reusing this project's own GraFT-inspired GraphEncoder (see
graph_encoder.py) -- previously found unstable when it replaced the flat
MLP encoder on the TRANSFORMER (30 vs 80 epoch runs gave wildly divergent
AUC on cic_ids2017, sometimes anti-correlated with truth). That instability
was hypothesized to be GNN oversmoothing/init-sensitivity, independent of
which transition mechanism sits downstream -- this is the direct test of
that hypothesis on a second, unrelated transition mechanism (RSSM's GRU +
stochastic latent instead of the Transformer's attention).

Everything else -- RSSM's own loss, the open-loop multi-step term, free-bits
KL -- is identical to train_rssm.py. Only the encoder changes.
"""
from __future__ import annotations

import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_datasets  # noqa: E402
from rssm import RSSM, RSSMConfig  # noqa: E402
from train_rssm import rssm_loss, evaluate  # noqa: E402

CONTEXT = 16
HORIZON = 5
CKPT_DIR = "/media/kavinder/hdd2/sih26153-processed/checkpoints"
CKPT_NAME = "rssm_graph_phase1_best.pt"


def main(epochs=30, batch_size=128, lr=3e-4, d_h=32, d_z=16, d_embed=64, d_hidden=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    datasets, scaler, _ = build_datasets(CONTEXT, HORIZON)
    print({role: len(ds) for role, ds in datasets.items()})
    loaders = {role: DataLoader(ds, batch_size=batch_size, shuffle=(role == "train"),
                                drop_last=(role == "train"))
              for role, ds in datasets.items()}

    cfg = RSSMConfig(n_features=len(scaler.center), d_h=d_h, d_z=d_z,
                     d_embed=d_embed, d_hidden=d_hidden, use_graph_encoder=True)
    model = RSSM(cfg).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}  (graph encoder ON)")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, CKPT_NAME)
    best_val = float("inf")

    for epoch in range(epochs):
        t0 = time.time()
        for batch in loaders["train"]:
            batch = batch.to(device)
            loss = rssm_loss(model, batch, CONTEXT)
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
                       "context": CONTEXT, "horizon": HORIZON, "arch": "rssm_graph"},
                      ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    test_loss = evaluate(model, loaders["test"], device, CONTEXT)
    print(f"best val_loss={best_val:.4f}   test_loss={test_loss:.4f}")
    print(f"checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
