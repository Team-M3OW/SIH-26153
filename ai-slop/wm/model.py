"""The world model: encoder -> distributional latent transition -> decoder.

No task heads here (stage classifier, infiltration head) -- those are Phase
2, trained separately on top of this once it can predict the future on its
own. This file only has to answer one question well: given z_1..z_t, what is
p(z_{t+1})? Everything downstream depends on that distribution being honest.

    z_t          = Encoder(S_t)
    mu_t, lv_t   = Transition(z_1..z_t)         # p(z_{t+1}) = N(mu_t, exp(lv_t))
    z_{t+1}      = mu_t + exp(0.5*lv_t) * eps   # reparameterised sample
    S_hat_{t+1}  = Decoder(z_{t+1})

Trained by scoring S_hat_{t+1} against the *real* S_{t+1} -- never against
S_t. A model that only learned to reconstruct its own input would score
badly on this loss by construction, which is what keeps this a dynamics
model rather than an autoencoder that happens to see sequences.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_encoder import GraphEncoder  # noqa: E402


class _GradientReversal(torch.autograd.Function):
    """Identity on the forward pass; negates (and scales) the gradient on
    the backward pass. This is the entire mechanism behind domain-adversarial
    training: everything downstream of this layer (the domain classifier)
    trains normally to get better at its job, but the gradient it sends
    backward is flipped before it reaches the encoder -- so the encoder is
    pushed to make the domain classifier's job *harder*, i.e. to stop
    encoding "which dataset did this window come from" at all."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, lambd)


class DomainHead(nn.Module):
    """Predicts which source dataset a window came from, from the SAME
    latent z the dynamics model produces. Only ever trained through
    grad_reverse -- see train.py. Not used at inference; exists purely to
    shape the encoder during Phase-1 training."""

    def __init__(self, d_latent: int, n_domains: int):
        super().__init__()
        self.fc = nn.Linear(d_latent, n_domains)

    def forward(self, z: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        return self.fc(grad_reverse(z, lambd))


class KoopmanOperator(nn.Module):
    """Constrains the latent's own, context-free dynamics to a block-diagonal
    2D rotation-scaling structure with a bounded spectral radius (r < r_max),
    instead of letting a free-form network decide how errors compound over a
    multi-step rollout. Each 2x2 block is `r * [[cos,-sin],[sin,cos]]` --
    damping (r<1), persistence (r=1), or (if r_max>1) growth/oscillation, but
    always within a KNOWN range, giving a provable multi-step error bound
    rather than hoping training finds stable dynamics on its own. This is
    what Koopman Dreamer (arXiv:2607.19719) adds to DreamerV3's transition,
    and its own ablation shows the spectral constraint alone -- not the rest
    of the architecture -- is what buys long-horizon stability.

    Added to (not replacing) the Transformer's own context-dependent
    prediction: this operator supplies the stable "generic" component of the
    dynamics, the Transformer supplies context-aware refinement on top.

    r_max=1.0 and raw_r initialised high (sigmoid(4)~=0.98) means every block
    starts as a near-identity, slight contraction -- close to the plain
    residual/persistence behaviour this replaces, so early training isn't
    disrupted, while still guaranteeing r<1 (a decaying, not exploding,
    multi-step error bound) from step one.
    """

    def __init__(self, d_latent: int, r_max: float = 1.0):
        super().__init__()
        self.d_latent = d_latent
        self.n_blocks = d_latent // 2
        self.has_extra = d_latent % 2 == 1
        self.r_max = r_max
        self.theta = nn.Parameter(torch.randn(self.n_blocks) * 0.1)
        self.raw_r = nn.Parameter(torch.full((self.n_blocks,), 4.0))
        if self.has_extra:
            self.raw_r_extra = nn.Parameter(torch.tensor(4.0))

    def _matrix(self, device, dtype) -> torch.Tensor:
        r = self.r_max * torch.sigmoid(self.raw_r)
        cos_t, sin_t = torch.cos(self.theta), torch.sin(self.theta)
        row0 = torch.stack([r * cos_t, -r * sin_t], dim=-1)   # (n_blocks, 2)
        row1 = torch.stack([r * sin_t,  r * cos_t], dim=-1)   # (n_blocks, 2)
        blocks = torch.stack([row0, row1], dim=1)             # (n_blocks, 2, 2)
        A = torch.block_diag(*blocks.unbind(0))
        if self.has_extra:
            r_extra = self.r_max * torch.sigmoid(self.raw_r_extra)
            A = torch.block_diag(A, r_extra.reshape(1, 1))
        return A.to(device=device, dtype=dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        A = self._matrix(z.device, z.dtype)
        return z @ A.T


@dataclass
class WorldModelConfig:
    n_features: int = 43
    d_model: int = 128
    d_latent: int = 32
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    max_len: int = 64
    min_logvar: float = -8.0
    max_logvar: float = 4.0
    d_node: int = 8         # GraphEncoder's per-feature node embedding dim
    k_sim: int = 5          # GraphEncoder's dynamic top-k similarity edges
    koopman_r_max: float = 1.0   # KoopmanOperator's max spectral radius
    use_koopman: bool = True     # False reverts to the plain residual, the
                                 # actual validated baseline for ablations

    def to_dict(self) -> dict:
        return asdict(self)


class CausalBlock(nn.Module):
    """Pre-norm causal self-attention block, hands back its attention map.

    Written explicitly rather than nn.TransformerEncoderLayer so the
    per-step attention weights are retrievable later -- they're the
    temporal explainability channel ("which past windows drove this").
    """

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                          dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, attn_mask, need_weights=False):
        h = self.ln1(x)
        a, w = self.attn(h, h, h, attn_mask=attn_mask,
                         need_weights=need_weights, average_attn_weights=True)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x, w


class WorldModel(nn.Module):
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg

        # Reverted to the flat MLP encoder: the R-GCN GraphEncoder (still
        # available in graph_encoder.py) proved unstable across training
        # lengths (see docs/architecture.md) and we're isolating the
        # Koopman-constrained transition as its own, separately-attributed
        # change on top of the known-stable baseline, not confounded with
        # the still-unresolved encoder question.
        self.encoder = nn.Sequential(
            nn.Linear(cfg.n_features, cfg.d_model), nn.LayerNorm(cfg.d_model),
            nn.GELU(), nn.Linear(cfg.d_model, cfg.d_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(cfg.d_latent, cfg.d_model), nn.LayerNorm(cfg.d_model),
            nn.GELU(), nn.Linear(cfg.d_model, cfg.n_features),
        )
        # Learned, per-feature, homoscedastic observation noise. Without it
        # the reconstruction/NLL loss is dominated by whichever features
        # happen to have the largest raw scale (e.g. n_flows vs port_entropy).
        self.obs_logvar = nn.Parameter(torch.zeros(cfg.n_features))

        self.z_in = nn.Linear(cfg.d_latent, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_len, cfg.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([CausalBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # Two outputs per latent dim (mean, log-variance) -- this is the
        # entire mechanism that makes the transition a distribution rather
        # than a point estimate. See docs/ discussion: "Gaussian head".
        self.trans_head = nn.Linear(cfg.d_model, 2 * cfg.d_latent)
        self.koopman = KoopmanOperator(cfg.d_latent, r_max=cfg.koopman_r_max)

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        return self.encoder(s)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def _causal_mask(self, t: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), 1)

    def transition(self, z_seq: torch.Tensor, need_weights: bool = False,
                  return_hidden: bool = False):
        """p(z_{t+1} | z_{<=t}) for every t in the sequence.

        Returns mu, logvar of shape (B, T, d_latent) -- position t predicts
        the latent at t+1. return_hidden=True also returns the pre-trans_head
        Transformer hidden state x -- used by the KL-bottleneck ablation to
        build a posterior on top without touching this method further.
        """
        t = z_seq.shape[1]
        x = self.z_in(z_seq) + self.pos[:, :t]
        mask = self._causal_mask(t, z_seq.device)
        attn = []
        for blk in self.blocks:
            x, w = blk(x, mask, need_weights=need_weights)
            if need_weights:
                attn.append(w)
        x = self.ln_f(x)
        mu_context, logvar = self.trans_head(x).chunk(2, dim=-1)
        logvar = logvar.clamp(self.cfg.min_logvar, self.cfg.max_logvar)
        if self.cfg.use_koopman:
            # mu = a spectrally-bounded, context-free Koopman step (the
            # "generic" dynamics, with a provable multi-step error bound)
            # PLUS the Transformer's own context-dependent refinement.
            mu = self.koopman(z_seq) + mu_context
        else:
            # Plain residual: predict the *change* in latent state -- the
            # actual validated baseline this ablation is isolated against.
            mu = mu_context + z_seq
        extra = (torch.stack(attn, 1) if need_weights else None)
        if return_hidden:
            return mu, logvar, extra, x
        return mu, logvar, extra

    @staticmethod
    def rsample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, s: torch.Tensor):
        """One teacher-forced pass: encode the whole sequence, predict every
        next step at once. Used for the Phase-1 pretraining loss."""
        z = self.encode(s)
        mu, logvar, _ = self.transition(z)
        return z, mu, logvar

    def imagine(self, s_ctx: torch.Tensor, k: int) -> torch.Tensor:
        """Differentiable open-loop rollout, one trajectory per input.

        Same open-loop rule as `rollout` (no real observation fed in past
        the context), but gradients flow through -- this is what lets a
        task head trained on these latents actually learn from the
        imagined path, rather than only ever seeing encodings of real data.
        """
        z = self.encode(s_ctx)
        zs = []
        for _ in range(k):
            mu, logvar, _ = self.transition(z[:, -self.cfg.max_len:])
            z_next = self.rsample(mu[:, -1], logvar[:, -1])
            zs.append(z_next)
            z = torch.cat([z, z_next.unsqueeze(1)], 1)
        return torch.stack(zs, 1)          # (B, k, d_latent)

    @torch.no_grad()
    def rollout(self, s_ctx: torch.Tensor, k: int, n_traj: int = 64,
               deterministic: bool = False):
        """Simulate k windows past the end of the observed context.

        No observation is fed in after the context ends -- each step
        consumes only the model's own sampled prediction. That open loop is
        what makes this a forecast rather than a detection: nothing from
        the horizon being predicted ever leaks into the input.

        Returns latents (B, K, n_traj, d_latent) and their decoded states
        (B, K, n_traj, n_features) -- task heads (Phase 2) get applied to
        the latents by whatever code trains them; this function knows
        nothing about stages or infiltration.
        """
        b, t, _ = s_ctx.shape
        z = self.encode(s_ctx)
        z = z.unsqueeze(1).expand(b, n_traj, t, -1).reshape(b * n_traj, t, -1)

        zs, decoded = [], []
        for _ in range(k):
            mu, logvar, _ = self.transition(z[:, -self.cfg.max_len:])
            mu, logvar = mu[:, -1], logvar[:, -1]
            z_next = mu if deterministic else self.rsample(mu, logvar)
            zs.append(z_next.view(b, n_traj, -1))
            decoded.append(self.decode(z_next).view(b, n_traj, -1))
            z = torch.cat([z, z_next.unsqueeze(1)], 1)

        return {
            "latents": torch.stack(zs, 1),          # (B, K, n_traj, d_latent)
            "decoded": torch.stack(decoded, 1),      # (B, K, n_traj, n_features)
        }


def gaussian_nll(target: torch.Tensor, mean: torch.Tensor,
                 logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * (logvar + (target - mean) ** 2 / logvar.exp())
