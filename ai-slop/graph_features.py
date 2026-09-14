"""Per-host communication-graph preprocessing for G-RSSM (arXiv:2604.14811,
Karacelebi et al.) -- nodes are network HOSTS (IP addresses), not features.
Only CTU-13 (Argus binetflow, SrcAddr/DstAddr) and CIC-IDS-2017 (raw CSVs,
Source IP/Destination IP) carry host identity; CIC-IDS-2018's public CSVs do
not, so this graph pipeline covers those two datasets only.

Per 60-second window (same WINDOW_SECONDS as the flat pipeline, so windows
line up by window_start with the existing labeled CSVs -- labels are never
recomputed here, only joined by timestamp):
  - up to MAX_NODES hosts, chosen by total flow count touching them
  - a small per-host feature vector (this host's own traffic behavior)
  - a binary adjacency among the selected hosts (did they communicate)
  - a mask marking real vs. zero-padded node slots

Deliberately NOT reusing IP as a model *feature* (see features.py's own
docstring on that) -- IP identity is used only to decide graph STRUCTURE
(who is a node, who is connected to whom), never fed as a value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_SECONDS = 60
MAX_NODES = 16   # tried 48 (wider roster, on the theory that CTU-13's
                 # secondary attack-relevant hosts rank outside top-16 by
                 # volume) -- made ctu13 no better and cic_ids2017 much
                 # worse (most windows only have a handful of genuinely
                 # active hosts even among 48 candidates, diluting signal
                 # with padding). Reverted; ctu13's real fix is still open.

NODE_FEATURE_NAMES = [
    "out_flow_count", "in_flow_count", "out_bytes_mean", "in_bytes_mean",
    "out_pkts_mean", "duration_mean", "distinct_partners", "distinct_dst_ports",
    "syn_rate", "proto_tcp_frac",
]
N_NODE_FEATURES = len(NODE_FEATURE_NAMES)


def _finalize_window(host_rows: pd.DataFrame, idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """host_rows: one window's flows, columns
    [src, dst, dport, is_tcp, dur, src_bytes, dst_bytes, src_pkts, syn].
    idx: FIXED host -> slot mapping, shared across every window in the same
    day (see build_graph_windows) -- node slot i must refer to the same
    host at every timestep, or the per-node GRU's recurrence is meaningless
    (this was the actual bug behind G-RSSM's anti-correlated forecasts:
    picking top hosts fresh per-window let slot i silently mean a
    different host from one minute to the next).
    Returns (node_feats (MAX_NODES, N_NODE_FEATURES), adjacency (MAX_NODES, MAX_NODES),
    mask (MAX_NODES,)) -- mask now means "this day-roster host was active
    THIS window", not "this host was in the top-K this window"."""
    active_hosts = set(host_rows["src"]).union(set(host_rows["dst"])) & idx.keys()

    node_feats = np.zeros((MAX_NODES, N_NODE_FEATURES), dtype=np.float32)
    adjacency = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
    mask = np.zeros((MAX_NODES,), dtype=np.float32)
    for h in active_hosts:
        mask[idx[h]] = 1.0

    for h in active_hosts:
        i = idx[h]
        as_src = host_rows[host_rows["src"] == h]
        as_dst = host_rows[host_rows["dst"] == h]
        partners = set(as_src["dst"]).union(set(as_dst["src"]))
        node_feats[i] = [
            len(as_src), len(as_dst),
            as_src["src_bytes"].mean() if len(as_src) else 0.0,
            as_dst["dst_bytes"].mean() if len(as_dst) else 0.0,
            as_src["src_pkts"].mean() if len(as_src) else 0.0,
            pd.concat([as_src["dur"], as_dst["dur"]]).mean() if (len(as_src) + len(as_dst)) else 0.0,
            len(partners),
            as_src["dport"].nunique() if len(as_src) else 0,
            as_src["syn"].mean() if len(as_src) else 0.0,
            pd.concat([as_src["is_tcp"], as_dst["is_tcp"]]).mean() if (len(as_src) + len(as_dst)) else 0.0,
        ]

    edges = host_rows[host_rows["src"].isin(idx) & host_rows["dst"].isin(idx)]
    for s, d in zip(edges["src"], edges["dst"]):
        i, j = idx[s], idx[d]
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0

    return node_feats, adjacency, mask


def build_graph_windows(df: pd.DataFrame, day_start: pd.Timestamp,
                        window_seconds: int = WINDOW_SECONDS) -> pd.DataFrame:
    """df: unified schema [timestamp, src, dst, dport, is_tcp, dur, src_bytes,
    dst_bytes, src_pkts, syn], already parsed (no NaT timestamps). Returns one
    row per window: window_start (unix seconds, int64) + node_feats/adjacency/
    mask as flattened columns (so the whole thing is a normal DataFrame,
    saved as .npz downstream for compactness)."""
    df = df.copy()
    df["window_id"] = ((df["timestamp"] - day_start).dt.total_seconds()
                       // window_seconds).astype(np.int64)
    df = df[df["window_id"] >= 0]

    # Fixed day-level host roster: top MAX_NODES hosts by total activity
    # across the WHOLE day, not re-picked per window -- so node slot i
    # refers to the same host at every timestep (see _finalize_window).
    out_deg = df.groupby("src").size()
    in_deg = df.groupby("dst").size()
    flow_count = out_deg.add(in_deg, fill_value=0)
    top_hosts = flow_count.sort_values(ascending=False).index[:MAX_NODES].tolist()
    idx = {h: i for i, h in enumerate(top_hosts)}

    wids = []
    node_feats_list, adjacency_list, mask_list = [], [], []
    for wid, group in df.groupby("window_id"):
        nf, adj, mask = _finalize_window(group, idx)
        wids.append(int(wid))
        node_feats_list.append(nf)
        adjacency_list.append(adj)
        mask_list.append(mask)

    return {
        # The absolute window index from day_start -- NOT a timestamp, to
        # sidestep any tz/serialization round-trip mismatch when joining
        # against the flat pipeline's own window_start column (which is
        # recomputed as the SAME kind of index there, not compared as an
        # epoch value -- see wm/graph_data.py).
        "window_id": np.array(wids, dtype=np.int64),
        "node_feats": np.stack(node_feats_list) if wids else np.zeros((0, MAX_NODES, N_NODE_FEATURES), np.float32),
        "adjacency": np.stack(adjacency_list) if wids else np.zeros((0, MAX_NODES, MAX_NODES), np.float32),
        "mask": np.stack(mask_list) if wids else np.zeros((0, MAX_NODES), np.float32),
    }
