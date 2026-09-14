"""A real RSSM (Recurrent State-Space Model, PlaNet/Dreamer lineage) --
built as a genuinely separate, directly-comparable alternative to model.py's
Transformer-based WorldModel, after two literature surveys this session
concluded full RSSM adoption isn't clearly justified for us (see
docs/literature_survey_rssm.md). Built anyway, empirically, isolated
cleanly, so the comparison rests on our own measurement rather than the
survey's reasoning alone.

State = [h_t ; z_t]:
  h_t = GRU(h_{t-1}, z_{t-1})                    deterministic recurrence,
                                                  never sees the current obs
  prior:     p(z_t | h_t)        = N(mu_p, sd_p)  used at inference/imagination
  posterior: q(z_t | h_t, S_t)   = N(mu_q, sd_q)  used at training, sees the
                                                  REAL current observation

Trained with DreamerV3's exact free-bits KL-balancing loss (arXiv:2301.04104):
  L_dyn = max(1, KL[sg(post) || prior])
  L_rep = max(1, KL[post || sg(prior)])
This exists specifically to stop posterior and prior collapsing onto each
other trivially -- a failure mode that can ONLY happen because a separate
posterior exists at all (see the survey). Implemented faithfully anyway,
since this is the actual mechanism being tested empirically.

Public interface (encode/decode/imagine/rollout/forward) deliberately
mirrors model.WorldModel so Phase 2 (train_heads.py) and the comparison
script can run unmodified against either architecture.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph_encoder import GraphEncoder  # noqa: E402


@dataclass
class RSSMConfig:
    n_features: int = 43
    d_h: int = 32          # deterministic recurrent state
    d_z: int = 16          # stochastic latent
    d_embed: int = 64      # encoder output width (posterior input)
    d_hidden: int = 64     # MLP hidden width elsewhere
    min_logvar: float = -8.0
    max_logvar: float = 4.0
    use_graph_encoder: bool = False   # False = flat MLP, the validated baseline
    d_node: int = 8         # GraphEncoder's per-feature node embedding dim
    k_sim: int = 5          # GraphEncoder's dynamic top-k similarity edges

    @property
    def d_latent(self) -> int:
        """[h;z] combined width -- what Phase 2 heads actually see."""
        return self.d_h + self.d_z

    def to_dict(self) -> dict:
        return asdict(self)


def free_bits_kl(post_mu, post_logvar, prior_mu, prior_logvar, free_bits=1.0):
    """DreamerV3's exact balanced free-bits KL: two directions, each clipped
    to at least `free_bits` nats, one gradient-blocked toward the prior side
    and one toward the posterior side. This asymmetry is what stops training
    from either freezing the prior or letting the posterior drift arbitrarily
    far from it."""
    def kl(mu_a, lv_a, mu_b, lv_b):
        var_a, var_b = lv_a.exp(), lv_b.exp()
        return 0.5 * (lv_b - lv_a + (var_a + (mu_a - mu_b) ** 2) / var_b - 1)

    kl_dyn = kl(post_mu.detach(), post_logvar.detach(), prior_mu, prior_logvar)
    kl_rep = kl(post_mu, post_logvar, prior_mu.detach(), prior_logvar.detach())
    l_dyn = torch.clamp(kl_dyn.sum(-1), min=free_bits)
    l_rep = torch.clamp(kl_rep.sum(-1), min=free_bits)
    return l_dyn, l_rep


class RSSM(nn.Module):
    def __init__(self, cfg: RSSMConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.use_graph_encoder:
            # GraFT-inspired R-GCN encoder (see graph_encoder.py) -- same
            # (..., n_features) -> (..., d_embed) calling convention as the
            # flat MLP it replaces, so nothing else here needs to change.
            self.encoder = GraphEncoder(cfg.n_features, cfg.d_embed,
                                       d_node=cfg.d_node, k_sim=cfg.k_sim)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(cfg.n_features, cfg.d_hidden), nn.LayerNorm(cfg.d_hidden),
                nn.GELU(), nn.Linear(cfg.d_hidden, cfg.d_embed),
            )
        self.decoder = nn.Sequential(
            nn.Linear(cfg.d_h + cfg.d_z, cfg.d_hidden), nn.LayerNorm(cfg.d_hidden),
            nn.GELU(), nn.Linear(cfg.d_hidden, cfg.n_features),
        )
        self.obs_logvar = nn.Parameter(torch.zeros(cfg.n_features))

        self.gru = nn.GRUCell(cfg.d_z, cfg.d_h)
        self.prior_net = nn.Sequential(
            nn.Linear(cfg.d_h, cfg.d_hidden), nn.GELU(),
            nn.Linear(cfg.d_hidden, 2 * cfg.d_z),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(cfg.d_h + cfg.d_embed, cfg.d_hidden), nn.GELU(),
            nn.Linear(cfg.d_hidden, 2 * cfg.d_z),
        )

    def _split(self, out: torch.Tensor):
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar.clamp(self.cfg.min_logvar, self.cfg.max_logvar)

    @staticmethod
    def rsample(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def _init_state(self, b, device):
        h = torch.zeros(b, self.cfg.d_h, device=device)
        z = torch.zeros(b, self.cfg.d_z, device=device)
        return h, z

    def forward(self, s: torch.Tensor):
        """Teacher-forced pass over a real sequence (B, T, F): posterior
        sees every real S_t, prior never does. Returns per-step posterior
        [h;z] states (what Phase 2's "real encoded states" means here) and
        everything needed for the KL loss."""
        b, t, _ = s.shape
        h, z = self._init_state(b, s.device)
        e = self.encoder(s)

        states, post_mus, post_lvs, prior_mus, prior_lvs = [], [], [], [], []
        for i in range(t):
            h = self.gru(z, h)
            prior_mu, prior_lv = self._split(self.prior_net(h))
            post_mu, post_lv = self._split(self.posterior_net(torch.cat([h, e[:, i]], -1)))
            z = self.rsample(post_mu, post_lv)
            states.append(torch.cat([h, z], -1))
            post_mus.append(post_mu); post_lvs.append(post_lv)
            prior_mus.append(prior_mu); prior_lvs.append(prior_lv)

        return (torch.stack(states, 1), torch.stack(post_mus, 1), torch.stack(post_lvs, 1),
               torch.stack(prior_mus, 1), torch.stack(prior_lvs, 1))

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        """Real, posterior-based [h;z] states -- the RSSM analogue of
        WorldModel.encode. Same calling convention: (..., n_features) in,
        (..., d_latent) out."""
        states, *_ = self.forward(s)
        return states

    def decode(self, hz: torch.Tensor) -> torch.Tensor:
        return self.decoder(hz)

    def imagine(self, s_ctx: torch.Tensor, horizon: int) -> torch.Tensor:
        """Differentiable open-loop rollout: run the REAL posterior over the
        context (so h_C reflects everything observed), then `horizon` steps
        using ONLY the prior -- no real observation, exactly mirroring
        WorldModel.imagine's contract."""
        b, c, _ = s_ctx.shape
        h, z = self._init_state(b, s_ctx.device)
        e = self.encoder(s_ctx)
        for i in range(c):
            h = self.gru(z, h)
            post_mu, post_lv = self._split(self.posterior_net(torch.cat([h, e[:, i]], -1)))
            z = self.rsample(post_mu, post_lv)

        states = []
        for _ in range(horizon):
            h = self.gru(z, h)
            prior_mu, prior_lv = self._split(self.prior_net(h))
            z = self.rsample(prior_mu, prior_lv)
            states.append(torch.cat([h, z], -1))
        return torch.stack(states, 1)

    @torch.no_grad()
    def rollout(self, s_ctx: torch.Tensor, k: int, n_traj: int = 64, deterministic: bool = False):
        """Inference-time ensemble rollout, same contract as WorldModel.rollout."""
        b, c, _ = s_ctx.shape
        h, z = self._init_state(b, s_ctx.device)
        e = self.encoder(s_ctx)
        for i in range(c):
            h = self.gru(z, h)
            post_mu, post_lv = self._split(self.posterior_net(torch.cat([h, e[:, i]], -1)))
            z = self.rsample(post_mu, post_lv)

        h = h.unsqueeze(1).expand(b, n_traj, -1).reshape(b * n_traj, -1)
        z = z.unsqueeze(1).expand(b, n_traj, -1).reshape(b * n_traj, -1)

        zs, decoded = [], []
        for _ in range(k):
            h = self.gru(z, h)
            prior_mu, prior_lv = self._split(self.prior_net(h))
            z = prior_mu if deterministic else self.rsample(prior_mu, prior_lv)
            hz = torch.cat([h, z], -1)
            zs.append(hz.view(b, n_traj, -1))
            decoded.append(self.decode(hz).view(b, n_traj, -1))
        return {"latents": torch.stack(zs, 1), "decoded": torch.stack(decoded, 1)}
