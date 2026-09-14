"""G-RSSM (arXiv:2604.14811, Karacelebi et al., "Learning Ad Hoc Network
Dynamics via Graph-Structured World Models") -- per-HOST latent states with
cross-host multi-head attention inside the transition, not a per-feature
graph like graph_encoder.py. A "node" here is a network host (IP address);
see graph_features.py for how the per-window host-communication graph
(up to MAX_NODES hosts, node features, adjacency, active-node mask) is
built from CTU-13/CIC-IDS-2017 raw flow records.

Per node i, per timestep t:
  h_t^i = GRU(z_{t-1}^i, h_{t-1}^i)              deterministic, own history
  h_t^i <- h_t^i + MaskedGraphAttention(h_t^*)   cross-node mixing, respects
                                                  this window's actual
                                                  communication adjacency
                                                  (never attends to padding
                                                  or unconnected hosts)
  prior:     p(z_t^i | h_t^i)
  posterior: q(z_t^i | h_t^i, obs_t^i)           sees this host's own
                                                  real features

Same free-bits KL as rssm.py (DreamerV3, arXiv:2301.04104), computed per
node and masked-averaged over active nodes only.

Public per-window-sequence interface (encode/imagine/rollout) returns a
POOLED flat (..., d_latent) vector -- a masked mean over active nodes'
[h;z] -- so Phase 2 heads (StageHead/InfilHead, unchanged) can consume it
exactly like model.WorldModel/rssm.RSSM's output. Internal, per-node
methods (forward/decode_nodes) are what train_grssm.py's loss actually
uses -- pooling only happens at the public boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn


@dataclass
class GRSSMConfig:
    n_node_features: int = 10
    max_nodes: int = 16
    d_h: int = 24           # per-node deterministic state
    d_z: int = 12           # per-node stochastic latent
    d_embed: int = 32       # per-node observation embedding
    d_hidden: int = 48
    n_heads: int = 4
    min_logvar: float = -8.0
    max_logvar: float = 4.0

    @property
    def d_latent(self) -> int:
        """Pooled [h;z] width -- what Phase 2 heads see, same convention
        as model.WorldModelConfig/rssm.RSSMConfig."""
        return self.d_h + self.d_z

    def to_dict(self) -> dict:
        return asdict(self)


def free_bits_kl(post_mu, post_logvar, prior_mu, prior_logvar, free_bits=1.0):
    """Same DreamerV3 balanced free-bits KL as rssm.py, unmodified."""
    def kl(mu_a, lv_a, mu_b, lv_b):
        var_a, var_b = lv_a.exp(), lv_b.exp()
        return 0.5 * (lv_b - lv_a + (var_a + (mu_a - mu_b) ** 2) / var_b - 1)

    kl_dyn = kl(post_mu.detach(), post_logvar.detach(), prior_mu, prior_logvar)
    kl_rep = kl(post_mu, post_logvar, prior_mu.detach(), prior_logvar.detach())
    l_dyn = torch.clamp(kl_dyn.sum(-1), min=free_bits)
    l_rep = torch.clamp(kl_rep.sum(-1), min=free_bits)
    return l_dyn, l_rep


class MaskedGraphAttention(nn.Module):
    """Multi-head attention among nodes in ONE window, restricted to actual
    communication edges (+ self-loops), never attending to padded/inactive
    node slots. Operates on a flat batch of (any leading dims, n_nodes, d)."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads, self.d_head = n_heads, d // n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """h: (N, n, d). adjacency: (N, n, n) binary. mask: (N, n) binary.
        Returns (N, n, d)."""
        N, n, d = h.shape
        q = self.q(h).view(N, n, self.n_heads, self.d_head).transpose(1, 2)  # (N,H,n,dh)
        k = self.k(h).view(N, n, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(h).view(N, n, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-1, -2)) / (self.d_head ** 0.5)          # (N,H,n,n)
        eye = torch.eye(n, device=h.device).unsqueeze(0)
        allow = ((adjacency + eye) > 0) & mask.unsqueeze(1).bool() & mask.unsqueeze(2).bool()
        allow = allow.unsqueeze(1)                                         # (N,1,n,n)
        scores = scores.masked_fill(~allow, -1e9)
        # A node with zero allowed neighbors (fully masked-out row) would
        # softmax a row of all -1e9 into uniform garbage -- pin those rows
        # to attend only to self so they at least pass through unchanged.
        no_neighbors = ~allow.any(dim=-1, keepdim=True)
        scores = torch.where(no_neighbors, torch.zeros_like(scores) + eye.unsqueeze(1) * 1e9
                             - (1 - eye.unsqueeze(1)) * 1e9, scores)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(N, n, d)
        return self.out(out)


class GRSSM(nn.Module):
    def __init__(self, cfg: GRSSMConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = nn.Sequential(
            nn.Linear(cfg.n_node_features, cfg.d_hidden), nn.LayerNorm(cfg.d_hidden),
            nn.GELU(), nn.Linear(cfg.d_hidden, cfg.d_embed),
        )
        self.decoder = nn.Sequential(
            nn.Linear(cfg.d_h + cfg.d_z, cfg.d_hidden), nn.LayerNorm(cfg.d_hidden),
            nn.GELU(), nn.Linear(cfg.d_hidden, cfg.n_node_features),
        )
        self.obs_logvar = nn.Parameter(torch.zeros(cfg.n_node_features))

        self.gru = nn.GRUCell(cfg.d_z, cfg.d_h)
        self.attn = MaskedGraphAttention(cfg.d_h, cfg.n_heads)
        self.prior_net = nn.Sequential(
            nn.Linear(cfg.d_h, cfg.d_hidden), nn.GELU(),
            nn.Linear(cfg.d_hidden, 2 * cfg.d_z),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(cfg.d_h + cfg.d_embed, cfg.d_hidden), nn.GELU(),
            nn.Linear(cfg.d_hidden, 2 * cfg.d_z),
        )

    def _split(self, out):
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar.clamp(self.cfg.min_logvar, self.cfg.max_logvar)

    @staticmethod
    def rsample(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def _init_state(self, b, n, device):
        h = torch.zeros(b, n, self.cfg.d_h, device=device)
        z = torch.zeros(b, n, self.cfg.d_z, device=device)
        return h, z

    def _step_h(self, z_prev, h_prev, adjacency_t, mask_t):
        b, n, _ = z_prev.shape
        h = self.gru(z_prev.reshape(b * n, -1), h_prev.reshape(b * n, -1)).view(b, n, -1)
        h = h + self.attn(h, adjacency_t, mask_t)
        return h

    def _pool(self, h, z, mask):
        """Masked mean over active nodes -> flat (..., d_latent)."""
        hz = torch.cat([h, z], -1)                          # (..., n, d_latent)
        m = mask.unsqueeze(-1)
        denom = m.sum(dim=-2).clamp(min=1.0)
        return (hz * m).sum(dim=-2) / denom

    def forward(self, node_feats: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor):
        """Teacher-forced pass over a real sequence.
        node_feats: (B,T,n,F). adjacency: (B,T,n,n). mask: (B,T,n).
        Returns per-node (pooled at the call site if needed): states (B,T,n,d_latent),
        post_mu, post_lv, prior_mu, prior_lv (all (B,T,n,d_z))."""
        b, t, n, _ = node_feats.shape
        device = node_feats.device
        e = self.encoder(node_feats)                        # (B,T,n,d_embed)
        h, z = self._init_state(b, n, device)

        states, post_mus, post_lvs, prior_mus, prior_lvs = [], [], [], [], []
        for i in range(t):
            h = self._step_h(z, h, adjacency[:, i], mask[:, i])
            prior_mu, prior_lv = self._split(self.prior_net(h))
            post_mu, post_lv = self._split(self.posterior_net(torch.cat([h, e[:, i]], -1)))
            z = self.rsample(post_mu, post_lv)
            states.append(torch.cat([h, z], -1))
            post_mus.append(post_mu); post_lvs.append(post_lv)
            prior_mus.append(prior_mu); prior_lvs.append(prior_lv)

        return (torch.stack(states, 1), torch.stack(post_mus, 1), torch.stack(post_lvs, 1),
               torch.stack(prior_mus, 1), torch.stack(prior_lvs, 1))

    def decode_nodes(self, states: torch.Tensor) -> torch.Tensor:
        return self.decoder(states)

    def encode(self, node_feats: torch.Tensor, adjacency: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
        """Real, posterior-based pooled states -- (B,T,d_latent)."""
        states, _, _, _, _ = self.forward(node_feats, adjacency, mask)
        return self._pool(states[..., :self.cfg.d_h], states[..., self.cfg.d_h:], mask)

    def imagine(self, node_feats_ctx: torch.Tensor, adjacency_ctx: torch.Tensor,
               mask_ctx: torch.Tensor, adjacency_future: torch.Tensor,
               mask_future: torch.Tensor, k: int) -> torch.Tensor:
        """Posterior over the context, then PRIOR-ONLY for k horizon steps
        -- differentiable, used by the multi-step training loss. Future
        adjacency/mask are still needed (graph structure isn't part of what
        the model predicts) -- same limitation the paper's own setup has
        for topology that isn't itself a random variable here."""
        b, t, n, _ = node_feats_ctx.shape
        device = node_feats_ctx.device
        e = self.encoder(node_feats_ctx)
        h, z = self._init_state(b, n, device)
        for i in range(t):
            h = self._step_h(z, h, adjacency_ctx[:, i], mask_ctx[:, i])
            post_mu, post_lv = self._split(self.posterior_net(torch.cat([h, e[:, i]], -1)))
            z = self.rsample(post_mu, post_lv)

        pooled = []
        for i in range(k):
            h = self._step_h(z, h, adjacency_future[:, i], mask_future[:, i])
            prior_mu, prior_lv = self._split(self.prior_net(h))
            z = self.rsample(prior_mu, prior_lv)
            pooled.append(self._pool(h, z, mask_future[:, i]))
        return torch.stack(pooled, 1)

    @torch.no_grad()
    def rollout(self, node_feats_ctx: torch.Tensor, adjacency_ctx: torch.Tensor,
               mask_ctx: torch.Tensor, adjacency_future: torch.Tensor,
               mask_future: torch.Tensor, k: int, n_traj: int = 64) -> dict:
        """No-grad ensemble rollout -- mirrors RSSM.rollout's contract,
        returns {"latents": (B,K,n_traj,d_latent)} of pooled states."""
        b, t, n, _ = node_feats_ctx.shape
        device = node_feats_ctx.device
        e = self.encoder(node_feats_ctx)
        h, z = self._init_state(b, n, device)
        for i in range(t):
            h = self._step_h(z, h, adjacency_ctx[:, i], mask_ctx[:, i])
            post_mu, post_lv = self._split(self.posterior_net(torch.cat([h, e[:, i]], -1)))
            z = self.rsample(post_mu, post_lv)

        h = h.unsqueeze(1).expand(b, n_traj, n, -1).reshape(b * n_traj, n, -1)
        z = z.unsqueeze(1).expand(b, n_traj, n, -1).reshape(b * n_traj, n, -1)
        adj_f = adjacency_future.unsqueeze(1).expand(b, n_traj, k, n, n).reshape(b * n_traj, k, n, n)
        mask_f = mask_future.unsqueeze(1).expand(b, n_traj, k, n).reshape(b * n_traj, k, n)

        pooled = []
        for i in range(k):
            h = self._step_h(z, h, adj_f[:, i], mask_f[:, i])
            prior_mu, prior_lv = self._split(self.prior_net(h))
            z = self.rsample(prior_mu, prior_lv)
            p = self._pool(h, z, mask_f[:, i]).view(b, n_traj, -1)
            pooled.append(p)
        return {"latents": torch.stack(pooled, 1)}
