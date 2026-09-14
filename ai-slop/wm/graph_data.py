"""Loads the per-host graph .npz files (see ../graph_features.py,
../preprocess_graph.py) and joins them to the existing flat CSVs' labels by
absolute window_id -- NOT by window_start string/epoch, which round-trips
through pandas/numpy differently enough between the two pipelines to be
unreliable (see session history: verified this the hard way). CTU-13's
day_start is its raw first-row timestamp (no midnight rounding); CIC-2017's
is that timestamp normalized to midnight -- both MUST match
features_ctu13.py/features_cicids2017.py exactly or window boundaries shift.

Only CTU-13 and CIC-IDS-2017 are covered; CIC-IDS-2018's public CSVs have no
IP columns so no host graph exists for it.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph_features import MAX_NODES, N_NODE_FEATURES  # noqa: E402

GRAPH_DIRS = {
    "ctu13": "/media/kavinder/hdd2/sih26153-processed/graph_ctu13",
    "cicids2017": "/media/kavinder/hdd2/sih26153-processed/graph_cicids2017",
}
FLAT_DIRS = {
    "ctu13": "/media/kavinder/hdd2/sih26153-processed/ctu13",
    "cicids2017": "/media/kavinder/hdd2/sih26153-processed/cicids2017",
}


def _flat_window_id(ts: pd.Series, family: str) -> np.ndarray:
    day_start = ts.iloc[0] if family == "ctu13" else ts.iloc[0].normalize()
    return ((ts - day_start).dt.total_seconds() // 60).astype(np.int64).values


def load_graph_days() -> dict[str, dict]:
    """Returns {name: {"node_feats", "adjacency", "mask", "stage_idx",
    "infiltration"}} -- graph windows inner-joined to flat labels by
    window_id, sorted by window_id (so sequence order is time order)."""
    out = {}
    for family, gdir in GRAPH_DIRS.items():
        fdir = FLAT_DIRS[family]
        for gpath in sorted(glob.glob(os.path.join(gdir, "*.graph.npz"))):
            name = os.path.basename(gpath).replace(".graph.npz", "")
            fpath = os.path.join(fdir, f"{name}.features.csv")
            if not os.path.exists(fpath):
                continue
            g = np.load(gpath)
            flat = pd.read_csv(fpath)
            ts = pd.to_datetime(flat["window_start"])
            flat_wid = _flat_window_id(ts, family)
            flat_lookup = dict(zip(flat_wid.tolist(),
                                   zip(flat["stage_idx"].values, flat["infiltration"].values)))

            gwid = g["window_id"]
            keep_idx, stage_idx, infil = [], [], []
            for i, wid in enumerate(gwid):
                lbl = flat_lookup.get(int(wid))
                if lbl is None:
                    continue
                keep_idx.append(i)
                stage_idx.append(lbl[0])
                infil.append(lbl[1])
            if not keep_idx:
                continue
            keep_idx = np.array(keep_idx)
            order = np.argsort(gwid[keep_idx])   # window_id is already ascending, but be explicit
            keep_idx = keep_idx[order]
            out[name] = {
                "node_feats": g["node_feats"][keep_idx],
                "adjacency": g["adjacency"][keep_idx],
                "mask": g["mask"][keep_idx],
                "stage_idx": np.array(stage_idx, dtype=np.int64)[order],
                "infiltration": np.array(infil, dtype=np.int64)[order],
            }
    return out


def temporal_split(day: dict, frac_train=0.70, frac_val=0.15) -> dict:
    n = len(day["stage_idx"])
    a, b = int(n * frac_train), int(n * (frac_train + frac_val))
    def sl(s):
        return {k: v[s] for k, v in day.items()}
    return {"train": sl(slice(0, a)), "val": sl(slice(a, b)), "test": sl(slice(b, n))}


class NodeFeatureScaler:
    """log1p (all 10 node features are non-negative counts/means/fractions)
    then median/IQR, fit on active (mask==1) nodes of the train split only
    -- same rationale as data.RobustScaler for the flat pipeline."""

    def fit(self, node_feats: np.ndarray, mask: np.ndarray) -> "NodeFeatureScaler":
        active = node_feats[mask.astype(bool)]           # (n_active_nodes, N_NODE_FEATURES)
        X = np.log1p(np.maximum(active, 0.0))
        self.center = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        iqr = q75 - q25
        self.scale = np.where(iqr < 1e-6, 1.0, iqr)
        return self

    def transform(self, node_feats: np.ndarray) -> np.ndarray:
        X = np.log1p(np.maximum(node_feats, 0.0))
        z = (X - self.center) / self.scale
        return np.clip(z, -8.0, 8.0).astype(np.float32)


class GraphSequenceDataset(Dataset):
    """Sliding (context+horizon) windows over one or more days' graph
    sequences. Item = (node_feats_seq (T,16,10), adjacency_seq (T,16,16),
    mask_seq (T,16)[, stage_idx_seq (T,), infiltration_seq (T,)])."""

    def __init__(self, days: list[dict], context: int, horizon: int, labeled: bool = False):
        self.days = days
        self.labeled = labeled
        self.span = context + horizon
        self.index = []
        for di, d in enumerate(days):
            n = len(d["stage_idx"])
            for start in range(0, n - self.span + 1):
                self.index.append((di, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        di, s = self.index[i]
        d = self.days[di]
        sl = slice(s, s + self.span)
        nf = torch.from_numpy(d["node_feats"][sl])
        adj = torch.from_numpy(d["adjacency"][sl])
        mask = torch.from_numpy(d["mask"][sl])
        if not self.labeled:
            return nf, adj, mask
        stage = torch.from_numpy(d["stage_idx"][sl]).long()
        infil = torch.from_numpy(d["infiltration"][sl]).long()
        return nf, adj, mask, stage, infil


def build_graph_datasets(context: int, horizon: int):
    """Unsupervised (Phase 1) datasets. Returns (datasets, scaler)."""
    raw = load_graph_days()
    if not raw:
        raise RuntimeError("no joined graph windows found -- did preprocess_graph.py "
                          "finish, and does every graph .npz have a matching flat CSV?")
    splits = {name: temporal_split(d) for name, d in raw.items()}

    train_nf = np.concatenate([s["train"]["node_feats"] for s in splits.values()], axis=0)
    train_mask = np.concatenate([s["train"]["mask"] for s in splits.values()], axis=0)
    scaler = NodeFeatureScaler().fit(train_nf, train_mask)

    datasets = {}
    for role in ("train", "val", "test"):
        days = []
        for s in splits.values():
            d = s[role]
            if len(d["stage_idx"]) == 0:
                continue
            days.append({**d, "node_feats": scaler.transform(d["node_feats"])})
        datasets[role] = GraphSequenceDataset(days, context, horizon, labeled=False)
    return datasets, scaler, splits


def build_labeled_graph_datasets(context: int, horizon: int, scaler: NodeFeatureScaler,
                                 splits: dict):
    datasets = {}
    for role in ("train", "val", "test"):
        days = []
        for s in splits.values():
            d = s[role]
            if len(d["stage_idx"]) == 0:
                continue
            days.append({**d, "node_feats": scaler.transform(d["node_feats"])})
        datasets[role] = GraphSequenceDataset(days, context, horizon, labeled=True)
    return datasets
