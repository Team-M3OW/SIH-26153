"""Phase 1 for the RSSM: teacher-forced obs+KL loss over the real sequence,
plus the same open-loop multi-step imagination loss already validated for
the Transformer model -- so both architectures are held to the identical
"must stay accurate over its own imagined rollout" standard, not just
one-step teacher forcing.

No stage/infiltration label touched here, same as train.py.
"""
from __future__ import annotations

import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_datasets  # noqa: E402
from model import gaussian_nll  # noqa: E402
from rssm import RSSM, RSSMConfig, free_bits_kl  # noqa: E402

CONTEXT = 16
HORIZON = 5
CKPT_DIR = "/media/kavinder/hdd2/sih26153-processed/checkpoints"
CKPT_NAME = "rssm_phase1_best.pt"


def rssm_loss(model: RSSM, batch: torch.Tensor, context: int) -> torch.Tensor:
    # Teacher-forced: posterior sees every real step, scored two ways --
    # reconstruction (decode the posterior's own [h;z]) and the balanced
    # free-bits KL pulling the observation-blind prior toward it.
    states, post_mu, post_lv, prior_mu, prior_lv = model(batch)
    s_hat = model.decode(states)
    obs_nll = gaussian_nll(batch, s_hat, model.obs_logvar.expand_as(s_hat))
    l_dyn, l_rep = free_bits_kl(post_mu, post_lv, prior_mu, prior_lv)
    teacher_forced = obs_nll.sum(-1) + l_dyn + l_rep     # (B, T)

    # Open-loop: real posterior over the context, then PRIOR-ONLY for the
    # horizon -- no KL here (no posterior to compare against once we've
    # stopped observing), scored the same way our Transformer model's
    # multi-step term is: decode and score against the real future.
    horizon = batch.shape[1] - context
    z_imagined = model.imagine(batch[:, :context], horizon)
    s_hat_multi = model.decode(z_imagined)
    target_multi = batch[:, context:context + horizon]
    nll_multi = gaussian_nll(target_multi, s_hat_multi,
                             model.obs_logvar.expand_as(s_hat_multi)).sum(-1)

    return torch.cat([teacher_forced.reshape(-1), nll_multi.reshape(-1)]).mean()


@torch.no_grad()
def evaluate(model: RSSM, loader: DataLoader, device: str, context: int) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        total += rssm_loss(model, batch, context).item() * len(batch)
        n += len(batch)
    model.train()
    return total / n


def main(epochs=30, batch_size=128, lr=3e-4, d_h=32, d_z=16, d_embed=64, d_hidden=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    datasets, scaler, _ = build_datasets(CONTEXT, HORIZON)
    print({role: len(ds) for role, ds in datasets.items()})
    loaders = {role: DataLoader(ds, batch_size=batch_size, shuffle=(role == "train"),
                                drop_last=(role == "train"))
              for role, ds in datasets.items()}

    cfg = RSSMConfig(n_features=len(scaler.center), d_h=d_h, d_z=d_z,
                     d_embed=d_embed, d_hidden=d_hidden)
    model = RSSM(cfg).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
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
                       "context": CONTEXT, "horizon": HORIZON, "arch": "rssm"},
                      ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    test_loss = evaluate(model, loaders["test"], device, CONTEXT)
    print(f"best val_loss={best_val:.4f}   test_loss={test_loss:.4f}")
    print(f"checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
