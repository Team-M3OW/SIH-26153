"""Ablation: isolate whether the KL-bottleneck mechanism itself (a posterior
that sees the real next observation, KL-balanced against a prior that
doesn't) is what made the RSSM better -- independent of GRU-vs-Transformer
backbone or parameter count.

Everything about the validated Transformer WorldModel stays EXACTLY as it
was (deterministic encoder feeds the sequence, Transformer computes the
existing one-step + multi-step dynamics loss unchanged, use_koopman=False to
match the actual baseline being compared against). The only addition: an
auxiliary PosteriorHead that takes [Transformer hidden state at t, encoded
real S_{t+1}] and predicts a "corrected" latent, trained via DreamerV3's
free-bits KL against the existing prior (trans_head's own output) plus a
reconstruction term. If this alone closes most of the RSSM/Transformer gap,
the KL-bottleneck mechanism is the answer, not the GRU or param count. If it
doesn't, the gap is in the backbone/capacity, not this mechanism.

The posterior is auxiliary only -- it does not replace the deterministic
encoder for propagating the sequence itself. That keeps this a true ablation
(one mechanism added) rather than a second, different architecture change.
"""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_datasets  # noqa: E402
from model import WorldModel, WorldModelConfig, gaussian_nll  # noqa: E402
from rssm import free_bits_kl  # noqa: E402
from train import dynamics_loss  # noqa: E402

CONTEXT = 16
HORIZON = 5
CKPT_DIR = "/media/kavinder/hdd2/sih26153-processed/checkpoints"
CKPT_NAME = "kl_ablation_phase1_best.pt"


class PosteriorHead(nn.Module):
    def __init__(self, d_model: int, d_latent: int, d_hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + d_latent, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 2 * d_latent),
        )

    def forward(self, x_t: torch.Tensor, z_next_raw: torch.Tensor):
        mu, logvar = self.net(torch.cat([x_t, z_next_raw], -1)).chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 4.0)


def kl_bottleneck_loss(model: WorldModel, posterior: PosteriorHead,
                       batch: torch.Tensor) -> torch.Tensor:
    z = model.encode(batch)                                    # (B, T, d_latent), same encoder as always
    prior_mu, prior_lv, _, x = model.transition(z[:, :-1], return_hidden=True)  # unmodified prior + its hidden state, positions 0..T-2

    post_mu, post_lv = posterior(x, z[:, 1:])                   # sees the REAL next encoded state
    l_dyn, l_rep = free_bits_kl(post_mu, post_lv, prior_mu, prior_lv)

    z_post_sample = post_mu + torch.randn_like(post_mu) * torch.exp(0.5 * post_lv)
    s_hat = model.decode(z_post_sample)
    recon_nll = gaussian_nll(batch[:, 1:], s_hat, model.obs_logvar.expand_as(s_hat)).sum(-1)

    return (l_dyn + l_rep + recon_nll).mean()


@torch.no_grad()
def evaluate(model, posterior, loader, device, context) -> float:
    model.eval(); posterior.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        loss = dynamics_loss(model, batch, context) + kl_bottleneck_loss(model, posterior, batch)
        total += loss.item() * len(batch)
        n += len(batch)
    model.train(); posterior.train()
    return total / n


def main(epochs=30, batch_size=128, lr=3e-4, d_model=64, d_latent=16,
        n_layers=2, n_heads=4, d_ff=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    datasets, scaler, _ = build_datasets(CONTEXT, HORIZON)
    print({role: len(ds) for role, ds in datasets.items()})
    loaders = {role: DataLoader(ds, batch_size=batch_size, shuffle=(role == "train"),
                                drop_last=(role == "train"))
              for role, ds in datasets.items()}

    cfg = WorldModelConfig(n_features=len(scaler.center), d_model=d_model,
                          d_latent=d_latent, n_layers=n_layers, n_heads=n_heads,
                          d_ff=d_ff, max_len=CONTEXT + HORIZON + 4, use_koopman=False)
    model = WorldModel(cfg).to(device)
    posterior = PosteriorHead(d_model, d_latent).to(device)
    print(f"params: model={sum(p.numel() for p in model.parameters()):,}  "
          f"posterior={sum(p.numel() for p in posterior.parameters()):,} (auxiliary, discarded after Phase 1)")
    opt = torch.optim.AdamW(list(model.parameters()) + list(posterior.parameters()),
                            lr=lr, weight_decay=1e-4)

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, CKPT_NAME)
    best_val = float("inf")

    for epoch in range(epochs):
        t0 = time.time()
        for batch in loaders["train"]:
            batch = batch.to(device)
            loss = dynamics_loss(model, batch, CONTEXT) + kl_bottleneck_loss(model, posterior, batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(posterior.parameters()), 1.0)
            opt.step()
        val_loss = evaluate(model, posterior, loaders["val"], device, CONTEXT)
        print(f"epoch {epoch+1:>3}/{epochs}  val_loss={val_loss:8.4f}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                       "scaler_center": torch.from_numpy(scaler.center),
                       "scaler_scale": torch.from_numpy(scaler.scale),
                       "context": CONTEXT, "horizon": HORIZON,
                       "arch": "transformer_kl_ablation"},  # distinct from "transformer" so
                       # Phase 2 doesn't overwrite the real baseline's phase2_head.pt
                      ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    test_loss = evaluate(model, posterior, loaders["test"], device, CONTEXT)
    print(f"best val_loss={best_val:.4f}   test_loss={test_loss:.4f}")
    print(f"checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
