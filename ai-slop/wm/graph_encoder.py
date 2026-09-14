"""A GraFT-inspired encoder: the 43 input features are treated as graph
nodes (one per window, per timestep), related by two edge types with
SEPARATE learned weights (a proper Relational GCN, not a single shared
adjacency multiply) -- replacing the flat MLP encoder entirely.

Scoped down from GraFT's actual four-relation-type design (see the AAAI'26
paper/repo): GraFT's Type-0 (temporal, same feature across time) and Type-2
(lagged cross-feature) edges are deliberately NOT reproduced here, because
our causal Transformer transition already owns cross-time modeling end to
end -- duplicating that inside the encoder would be redundant risk for no
established benefit. What's kept is GraFT's core, most novel-to-us idea:
explicit *relational* structure among the features within one window,
processed by relation-specific weights, instead of one undifferentiated
fully-connected layer.

Two relation types, both operating within a single timestep:
  static  -- our own domain-knowledge feature groupings (TCP-flag features
             relate to each other, port-structure features relate to each
             other, etc.) -- fixed, the same for every window, every
             dataset. This is what makes the relation itself
             domain-invariant by construction: "port entropy relates to
             distinct-port fraction" doesn't depend on which network
             produced the traffic.
  dynamic -- top-k cosine-similarity edges among the CURRENT window's own
             feature embeddings (GraFT's Type-3 idea) -- instance-specific,
             adapts per window rather than being fixed in advance.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import INPUT_FEATURE_COLUMNS  # noqa: E402

# Our own domain-knowledge groupings of the 43 features -- fully connected
# within each group for the static relation. Covers all 43 columns exactly
# once; see features.py for what each column means.
FEATURE_GROUPS = [
    ["syn_rate", "ack_rate", "fin_rate", "rst_rate", "psh_rate", "urg_rate"],
    ["port_entropy", "distinct_port_frac", "svc_ftp_frac", "svc_ssh_frac",
     "svc_telnet_frac", "svc_smtp_frac", "svc_dns_frac", "svc_http_frac",
     "svc_https_frac", "svc_smb_frac", "svc_rdp_frac"],
    ["proto_tcp_frac", "proto_udp_frac", "proto_other_frac"],
    ["n_flows", "fwd_pkts_mean", "bwd_pkts_mean", "fwd_bytes_mean", "bwd_bytes_mean",
     "byts_per_s_mean", "byts_per_s_std", "pkts_per_s_mean", "pkts_per_s_std"],
    ["duration_mean", "duration_std", "iat_mean_mean", "init_fwd_win_mean",
     "init_bwd_win_mean", "half_open_frac"],
    ["down_up_ratio_mean", "oneway_frac"],
    ["has_packet_level", "ttl_mean", "ttl_std", "frag_rate", "retrans_rate",
     "payload_len_std"],
]


def build_static_adjacency(feature_columns: list[str] = INPUT_FEATURE_COLUMNS) -> torch.Tensor:
    n = len(feature_columns)
    idx = {c: i for i, c in enumerate(feature_columns)}
    covered = {c for g in FEATURE_GROUPS for c in g}
    missing = set(feature_columns) - covered
    if missing:
        raise ValueError(f"FEATURE_GROUPS doesn't cover: {missing}")
    adj = np.zeros((n, n), dtype=np.float32)
    for group in FEATURE_GROUPS:
        ids = [idx[c] for c in group]
        for i in ids:
            for j in ids:
                if i != j:
                    adj[i, j] = 1.0
    return torch.from_numpy(adj)


class GraphEncoder(nn.Module):
    def __init__(self, n_features: int, d_latent: int, d_node: int = 8, k_sim: int = 5):
        super().__init__()
        self.n_features = n_features
        self.d_node = d_node
        self.k_sim = min(k_sim, n_features - 1)

        # Feature tokenization: each raw scalar -> its own learned d_node
        # vector (a per-feature affine transform), so message passing has
        # something richer than a bare number to work with.
        self.node_w = nn.Parameter(torch.randn(n_features, d_node) * 0.1)
        self.node_b = nn.Parameter(torch.zeros(n_features, d_node))

        self.register_buffer("static_adj", build_static_adjacency())

        # Relational GCN: one weight matrix per relation type, plus a
        # self-loop weight -- summed, not concatenated, following the
        # standard R-GCN recipe (Schlichtkrull et al. 2018 / GraFT's own
        # RelationalGCNLayer).
        self.w_self = nn.Linear(d_node, d_node, bias=False)
        self.w_static = nn.Linear(d_node, d_node, bias=False)
        self.w_dynamic = nn.Linear(d_node, d_node, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_node))
        self.act = nn.GELU()

        self.pool = nn.Linear(n_features * d_node, d_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., n_features) -- any number of leading (batch/time) dims,
        same calling convention as the flat MLP encoder it replaces."""
        lead_shape = x.shape[:-1]
        xf = x.reshape(-1, self.n_features)                     # (N, F)
        h = xf.unsqueeze(-1) * self.node_w + self.node_b         # (N, F, d_node)

        static_agg = torch.einsum("ij,njd->nid", self.static_adj, h)
        static_msg = self.w_static(static_agg)

        h_norm = F.normalize(h, p=2, dim=-1)
        sim = torch.bmm(h_norm, h_norm.transpose(1, 2))          # (N, F, F)
        sim = sim - torch.eye(self.n_features, device=x.device) * 1e9  # exclude self
        _, topk_idx = sim.topk(self.k_sim, dim=-1)
        dyn_adj = torch.zeros_like(sim).scatter_(-1, topk_idx, 1.0)
        dyn_agg = torch.bmm(dyn_adj, h)
        dyn_msg = self.w_dynamic(dyn_agg)

        h_out = self.act(self.w_self(h) + static_msg + dyn_msg + self.bias)  # (N, F, d_node)
        z = self.pool(h_out.reshape(h_out.shape[0], -1))         # (N, d_latent)
        return z.reshape(*lead_shape, -1)
