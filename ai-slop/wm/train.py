"""Phase 1: self-supervised dynamics pretraining.

One loss, computed against tomorrow's window, never today's -- but scored
two ways, concatenated into that one loss rather than weighted against each
other:

  one-step, teacher-forced: real z_{<=t} in, score z_{t+1} against reality,
      for every position. This is what the original version of this file
      trained on exclusively.
  multi-step, open-loop: encode only the first CONTEXT positions, roll the
      transition forward HORIZON steps consuming nothing but its own
      sampled output (model.imagine, differentiable), decode each imagined
      step, score against the REAL future observations at those positions.

Training on the one-step term alone never teaches the model to stay
accurate over an open-loop rollout -- it only ever sees real history at
every step during training, then gets asked to imagine chains of its own
predictions at inference. This is the fix: score the actual open-loop
rollout during training too (this is what "latent overshooting" means in
the RSSM literature), so the multi-step forecast the demo actually reports
is what the loss has been optimizing the whole time -- not a regime the
model has never been evaluated in until inference.

No stage/infiltration label is read or touched anywhere in this file --
that is Phase 2's job, on top of this.

Domain-adversarial training. Pooling three genuinely different network
environments (CIC-IDS-2018, CIC-IDS-2017, CTU-13) into one model caused a
real, measured failure: Pi and the stage classifier worked on whichever
domain dominated training and were actively worse than random on the
others. A per-dataset scaler would "fix" this only for the three datasets
we happen to have -- not a foundational fix. Domain-adversarial training is:
add a small classifier that guesses which dataset a window came from, and
train the ENCODER (via a gradient-reversal layer -- see model.py) to make
that classifier fail. What survives in z is (in principle) whatever's
useful for predicting dynamics *and* doesn't reveal which training network
produced it -- a representation actively pushed toward transferring to a
network the model has never seen, not fitted to the three it has.
"""
from __future__ import annotations

import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_datasets, DOMAIN_NAMES  # noqa: E402
from model import WorldModel, WorldModelConfig, DomainHead, gaussian_nll  # noqa: E402

CONTEXT = 16
HORIZON = 5
CKPT_DIR = "/media/kavinder/hdd2/sih26153-processed/checkpoints"


def dynamics_loss(model: WorldModel, batch: torch.Tensor, context: int) -> torch.Tensor:
    # one-step, teacher-forced -- every position predicts the real next step
    _, mu, logvar = model(batch)
    s_hat_1step = model.decode(mu[:, :-1])
    nll_1step = gaussian_nll(batch[:, 1:], s_hat_1step,
                             model.obs_logvar.expand_as(s_hat_1step))

    # multi-step, open-loop -- only the context is real; every step after
    # that consumes the model's own previous prediction, exactly as it will
    # at inference. Scored against the real observations at those positions.
    horizon = batch.shape[1] - context
    z_imagined = model.imagine(batch[:, :context], horizon)
    s_hat_multi = model.decode(z_imagined)
    target_multi = batch[:, context:context + horizon]
    nll_multi = gaussian_nll(target_multi, s_hat_multi,
                             model.obs_logvar.expand_as(s_hat_multi))

    # Concatenated into one loss, not weighted against each other -- both
    # terms are the same Gaussian NLL against real data, just from two
    # different rollout regimes.
    return torch.cat([nll_1step.reshape(-1), nll_multi.reshape(-1)]).mean()


def domain_loss(model: WorldModel, domain_head: DomainHead, feats: torch.Tensor,
                domain_ids: torch.Tensor, lambd: float) -> torch.Tensor:
    """Cross-entropy of the domain classifier against the true source
    dataset, computed at every context position (domain doesn't change
    within one window's sequence, so the same label applies throughout).
    grad_reverse (inside DomainHead) is what makes this adversarial for the
    encoder -- this function itself just computes an ordinary classification
    loss, same as any other head."""
    z = model.encode(feats)
    logits = domain_head(z, lambd)                      # (B, T, n_domains)
    targets = domain_ids.unsqueeze(1).expand(-1, logits.shape[1])
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def grl_lambda(progress: float) -> float:
    """Standard DANN ramp (Ganin & Lempitsky 2015): 0 at the start of
    training, smoothly approaching 1. Full adversarial strength from step
    one would fight the dynamics objective before the encoder has learned
    anything worth protecting; ramping lets the main task get a head start."""
    import math
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


@torch.no_grad()
def evaluate(model: WorldModel, loader: DataLoader, device: str, context: int) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch, _domain in loader:
        batch = batch.to(device)
        total += dynamics_loss(model, batch, context).item() * len(batch)
        n += len(batch)
    model.train()
    return total / n


def main(epochs=30, batch_size=128, lr=3e-4, d_model=64, d_latent=16,
        n_layers=2, n_heads=4, d_ff=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    datasets, scaler, _ = build_datasets(CONTEXT, HORIZON, include_domain=True)
    print({role: len(ds) for role, ds in datasets.items()})
    loaders = {role: DataLoader(ds, batch_size=batch_size, shuffle=(role == "train"),
                                drop_last=(role == "train"))
              for role, ds in datasets.items()}

    cfg = WorldModelConfig(n_features=len(scaler.center), d_model=d_model,
                          d_latent=d_latent, n_layers=n_layers, n_heads=n_heads,
                          d_ff=d_ff, max_len=CONTEXT + HORIZON + 4)
    model = WorldModel(cfg).to(device)
    domain_head = DomainHead(d_latent, len(DOMAIN_NAMES)).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,} "
          f"(+{sum(p.numel() for p in domain_head.parameters()):,} domain head, "
          f"discarded after Phase 1)")
    opt = torch.optim.AdamW(list(model.parameters()) + list(domain_head.parameters()),
                            lr=lr, weight_decay=1e-4)

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, "phase1_best.pt")
    best_val = float("inf")
    n_batches = len(loaders["train"])

    for epoch in range(epochs):
        t0 = time.time()
        for bi, (batch, domain_ids) in enumerate(loaders["train"]):
            batch, domain_ids = batch.to(device), domain_ids.to(device)
            progress = (epoch * n_batches + bi) / max(epochs * n_batches - 1, 1)
            lambd = grl_lambda(progress)

            loss = dynamics_loss(model, batch, CONTEXT)
            loss = loss + domain_loss(model, domain_head, batch[:, :CONTEXT], domain_ids, lambd)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(domain_head.parameters()), 1.0)
            opt.step()
        val_loss = evaluate(model, loaders["val"], device, CONTEXT)
        print(f"epoch {epoch+1:>3}/{epochs}  val_nll={val_loss:8.4f}  lambda={lambd:.3f}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                       "scaler_center": torch.from_numpy(scaler.center),
                       "scaler_scale": torch.from_numpy(scaler.scale),
                       "context": CONTEXT, "horizon": HORIZON},
                      ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    test_loss = evaluate(model, loaders["test"], device, CONTEXT)
    print(f"best val_nll={best_val:.4f}   test_nll={test_loss:.4f}")
    print(f"checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
