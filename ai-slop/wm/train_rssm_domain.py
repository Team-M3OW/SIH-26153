"""Phase 1 for the RSSM + domain-adversarial training, combined.

Domain-adversarial training (see train.py) was a partial fix for the
Transformer: it helped cic_ids2017 (the worst domain) but hurt ctu13
(previously the best). RSSM independently turned out to generalize far
better to cic_ids2017/2018 than either Transformer variant, but is weakest
on ctu13. Since the two techniques target the same failure (domain
generalization) through different mechanisms -- DANN by actively erasing
"which dataset" from the latent, RSSM by its KL-bottleneck regularizing
what the latent can retain at all -- this combines both to see whether they
stack, cancel, or whether RSSM's bottleneck makes the domain classifier's
job moot (nothing domain-specific survives to erase in the first place).

Everything about RSSM's own loss (teacher-forced obs+KL, plus the open-loop
multi-step imagination term) is unchanged from train_rssm.py. The only
addition: a DomainHead (identical to the Transformer's) applied to
model.encode()'s posterior [h;z] states through a gradient-reversal layer,
with the same DANN sigmoid ramp on lambda.
"""
from __future__ import annotations

import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_datasets, DOMAIN_NAMES  # noqa: E402
from model import DomainHead, gaussian_nll  # noqa: E402
from rssm import RSSM, RSSMConfig, free_bits_kl  # noqa: E402
from train import grl_lambda  # noqa: E402
from train_rssm import rssm_loss  # noqa: E402

CONTEXT = 16
HORIZON = 5
CKPT_DIR = "/media/kavinder/hdd2/sih26153-processed/checkpoints"
CKPT_NAME = "rssm_domain_phase1_best.pt"


def domain_loss(model: RSSM, domain_head: DomainHead, feats: torch.Tensor,
                domain_ids: torch.Tensor, lambd: float) -> torch.Tensor:
    z = model.encode(feats)                              # posterior [h;z], (B, T, d_latent)
    logits = domain_head(z, lambd)
    targets = domain_ids.unsqueeze(1).expand(-1, logits.shape[1])
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


@torch.no_grad()
def evaluate(model: RSSM, loader: DataLoader, device: str, context: int) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch, _domain in loader:
        batch = batch.to(device)
        total += rssm_loss(model, batch, context).item() * len(batch)
        n += len(batch)
    model.train()
    return total / n


def main(epochs=30, batch_size=128, lr=3e-4, d_h=32, d_z=16, d_embed=64, d_hidden=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    datasets, scaler, _ = build_datasets(CONTEXT, HORIZON, include_domain=True)
    print({role: len(ds) for role, ds in datasets.items()})
    loaders = {role: DataLoader(ds, batch_size=batch_size, shuffle=(role == "train"),
                                drop_last=(role == "train"))
              for role, ds in datasets.items()}

    cfg = RSSMConfig(n_features=len(scaler.center), d_h=d_h, d_z=d_z,
                     d_embed=d_embed, d_hidden=d_hidden)
    model = RSSM(cfg).to(device)
    domain_head = DomainHead(cfg.d_latent, len(DOMAIN_NAMES)).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,} "
          f"(+{sum(p.numel() for p in domain_head.parameters()):,} domain head, "
          f"discarded after Phase 1)")
    opt = torch.optim.AdamW(list(model.parameters()) + list(domain_head.parameters()),
                            lr=lr, weight_decay=1e-4)

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, CKPT_NAME)
    best_val = float("inf")
    n_batches = len(loaders["train"])

    for epoch in range(epochs):
        t0 = time.time()
        for bi, (batch, domain_ids) in enumerate(loaders["train"]):
            batch, domain_ids = batch.to(device), domain_ids.to(device)
            progress = (epoch * n_batches + bi) / max(epochs * n_batches - 1, 1)
            lambd = grl_lambda(progress)

            loss = rssm_loss(model, batch, CONTEXT)
            loss = loss + domain_loss(model, domain_head, batch[:, :CONTEXT], domain_ids, lambd)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(domain_head.parameters()), 1.0)
            opt.step()
        val_loss = evaluate(model, loaders["val"], device, CONTEXT)
        print(f"epoch {epoch+1:>3}/{epochs}  val_loss={val_loss:8.4f}  lambda={lambd:.3f}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                       "scaler_center": torch.from_numpy(scaler.center),
                       "scaler_scale": torch.from_numpy(scaler.scale),
                       "context": CONTEXT, "horizon": HORIZON, "arch": "rssm_domain"},
                      ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    test_loss = evaluate(model, loaders["test"], device, CONTEXT)
    print(f"best val_loss={best_val:.4f}   test_loss={test_loss:.4f}")
    print(f"checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
