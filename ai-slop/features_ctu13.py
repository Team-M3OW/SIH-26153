"""CTU-13 Argus binetflow CSV -> the same windowed feature schema as
features.py, so both datasets can be concatenated into one training pool.

Argus binetflow is a coarser, different flow record than CICFlowMeter:
- Proto/State are strings ("tcp"/"udp"/..., "CON"/"FIN"/"RST"/"REQ"/...),
  not flag counts -- there is no clean per-flag (SYN/ACK/PSH/URG) count to
  recover from State, only FIN and RST have a direct state-name match.
  syn_rate/ack_rate/psh_rate/urg_rate are zero-filled here, honestly, not
  approximated from something that doesn't actually mean the same thing.
- Dur is in SECONDS. CICFlowMeter's Flow Duration is in MICROSECONDS. This
  is a real unit mismatch, not just a rename -- get it wrong and every
  duration/rate feature is off by 1e6x between the two datasets, silently
  corrupting anything trained on both. Converted to microseconds here to
  match what features.py already produced.
- No -1 half-open sentinel convention on SrcWin/DstWin (that's a
  CICFlowMeter-specific encoding) -- half_open_frac is zero-filled.
- What CTU-13 has that CIC-IDS-2018's CSVs mostly don't: real Src/Dst IP
  (used only as an aggregation key, never a feature) and real source TTL
  (sTtl) -- ttl_mean/ttl_std are populated from real data here, not zeroed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from attack_map import stage_for_label, STAGE_TO_IDX, INFILTRATION_STAGES
from features import FEATURE_COLUMNS, _SERVICE_PORTS

WINDOW_SECONDS = 60
CHUNK_ROWS = 500_000


def _clean_numeric(s: pd.Series) -> pd.Series:
    return (pd.to_numeric(s, errors="coerce")
              .replace([np.inf, -np.inf], np.nan)
              .fillna(0.0))


def _parse_timestamp(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    bad = ts.isna() & s.notna()
    if bad.any():
        ts.loc[bad] = pd.to_datetime(s[bad], errors="coerce")
    return ts


def _prepare_chunk(chunk: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    dport = pd.to_numeric(chunk["Dport"], errors="coerce")
    ts = _parse_timestamp(chunk["StartTime"])
    keep = dport.notna() & ts.notna()
    chunk = chunk.loc[keep].copy()
    dport = dport.loc[keep]
    ts = ts.loc[keep]

    chunk["window_id"] = ((ts - t0).dt.total_seconds() // WINDOW_SECONDS).astype(np.int64)
    chunk["Dport"] = dport.astype(np.int64)

    proto = chunk["Proto"].astype(str).str.lower()
    chunk["is_tcp"] = (proto == "tcp").astype(np.float64)
    chunk["is_udp"] = (proto == "udp").astype(np.float64)
    chunk["is_other"] = (~proto.isin(["tcp", "udp"])).astype(np.float64)

    state = chunk["State"].astype(str)
    chunk["is_fin"] = state.str.contains("FIN", na=False).astype(np.float64)
    chunk["is_rst"] = state.str.contains("RST", na=False).astype(np.float64)

    dur_sec = _clean_numeric(chunk["Dur"])
    chunk["duration_us"] = dur_sec * 1e6            # match CICFlowMeter's units
    src_pkts = _clean_numeric(chunk["SrcPkts"])
    dst_pkts = _clean_numeric(chunk["DstPkts"])
    src_bytes = _clean_numeric(chunk["SrcBytes"])
    dst_bytes = _clean_numeric(chunk["DstBytes"])
    tot_pkts = src_pkts + dst_pkts
    tot_bytes = src_bytes + dst_bytes
    safe_dur = dur_sec.replace(0, np.nan)
    chunk["byts_per_s"] = (tot_bytes / safe_dur).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    chunk["pkts_per_s"] = (tot_pkts / safe_dur).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    chunk["down_up_ratio"] = (dst_bytes / src_bytes.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan).fillna(0.0)

    chunk["fwd_pkts"] = src_pkts
    chunk["bwd_pkts"] = dst_pkts
    chunk["fwd_bytes"] = src_bytes
    chunk["bwd_bytes"] = dst_bytes
    chunk["init_fwd_win"] = _clean_numeric(chunk["SrcWin"])
    chunk["init_bwd_win"] = _clean_numeric(chunk["DstWin"])
    chunk["oneway"] = (dst_bytes == 0).astype(np.float64)
    chunk["sttl"] = _clean_numeric(chunk["sTtl"])

    for col in ["duration_us", "byts_per_s", "pkts_per_s", "sttl"]:
        chunk[col + "_sq"] = chunk[col] ** 2
    return chunk


NUMERIC_SUM_COLS = [
    "duration_us", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes",
    "byts_per_s", "pkts_per_s", "down_up_ratio", "init_fwd_win", "init_bwd_win",
    "is_tcp", "is_udp", "is_other", "oneway", "is_fin", "is_rst", "sttl",
]
SQ_COLS = ["duration_us", "byts_per_s", "pkts_per_s", "sttl"]


def process_file(path: str, window_seconds: int = WINDOW_SECONDS,
                 chunk_rows: int = CHUNK_ROWS) -> pd.DataFrame:
    usecols = ["Proto", "Dport", "State", "StartTime", "SrcPkts", "DstPkts",
              "SrcBytes", "DstBytes", "Dur", "SrcWin", "DstWin", "sTtl", "Label"]

    t0 = None
    sums = None
    n_flows = None
    port_counts: dict[int, dict[int, int]] = {}
    label_counts: dict[int, dict[str, int]] = {}

    reader = pd.read_csv(path, usecols=usecols, chunksize=chunk_rows, low_memory=False)
    for raw_chunk in reader:
        if t0 is None:
            first_ts = _parse_timestamp(raw_chunk["StartTime"]).dropna().iloc[0]
            t0 = first_ts

        chunk = _prepare_chunk(raw_chunk, t0)
        if chunk.empty:
            continue

        agg_cols = NUMERIC_SUM_COLS + [c + "_sq" for c in SQ_COLS]
        chunk_sums = chunk.groupby("window_id")[agg_cols].sum()
        chunk_n = chunk.groupby("window_id").size()
        sums = chunk_sums if sums is None else sums.add(chunk_sums, fill_value=0.0)
        n_flows = chunk_n if n_flows is None else n_flows.add(chunk_n, fill_value=0.0)

        for (wid, port), cnt in chunk.groupby(["window_id", "Dport"]).size().items():
            port_counts.setdefault(wid, {})
            port_counts[wid][port] = port_counts[wid].get(port, 0) + int(cnt)
        for (wid, label), cnt in chunk.groupby(["window_id", "Label"]).size().items():
            label_counts.setdefault(wid, {})
            label_counts[wid][label] = label_counts[wid].get(label, 0) + int(cnt)

    if sums is None:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    rows = []
    for wid in sorted(sums.index.astype(int)):
        n = float(n_flows.loc[wid])
        s = sums.loc[wid]

        def mean(col):
            return float(s[col]) / n if n > 0 else 0.0

        def std(col):
            m = mean(col)
            var = float(s[col + "_sq"]) / n - m ** 2 if n > 0 else 0.0
            return float(np.sqrt(max(var, 0.0)))

        ports = port_counts.get(wid, {})
        total_port_obs = sum(ports.values()) or 1
        probs = np.array([c / total_port_obs for c in ports.values()])
        port_entropy = float(-(probs * np.log2(probs)).sum()) if len(probs) else 0.0
        distinct_port_frac = len(ports) / n if n > 0 else 0.0
        svc_fracs = {name: sum(ports.get(p, 0) for p in plist) / total_port_obs
                    for name, plist in _SERVICE_PORTS.items()}

        labels = label_counts.get(wid, {})
        n_attack = sum(c for lbl, c in labels.items()
                      if "normal" not in lbl.lower() and "background" not in lbl.lower())
        attack_frac = n_attack / n if n > 0 else 0.0
        non_benign = {lbl: c for lbl, c in labels.items()
                     if "normal" not in lbl.lower() and "background" not in lbl.lower()}
        dominant_label = max(non_benign.items(), key=lambda x: x[1])[0] if non_benign else \
            max(labels.items(), key=lambda x: x[1])[0]
        stage = stage_for_label(dominant_label)
        stage_idx = STAGE_TO_IDX.get(stage, -1)
        infiltration = int(stage in INFILTRATION_STAGES)

        row = {
            "window_start": (t0 + pd.Timedelta(seconds=wid * window_seconds)).isoformat(),
            "n_flows": n,
            "duration_mean": mean("duration_us"), "duration_std": std("duration_us"),
            "fwd_pkts_mean": mean("fwd_pkts"), "bwd_pkts_mean": mean("bwd_pkts"),
            "fwd_bytes_mean": mean("fwd_bytes"), "bwd_bytes_mean": mean("bwd_bytes"),
            "byts_per_s_mean": mean("byts_per_s"), "byts_per_s_std": std("byts_per_s"),
            "pkts_per_s_mean": mean("pkts_per_s"), "pkts_per_s_std": std("pkts_per_s"),
            "iat_mean_mean": 0.0,               # not available in Argus binetflow
            "down_up_ratio_mean": mean("down_up_ratio"),
            "init_fwd_win_mean": mean("init_fwd_win"), "init_bwd_win_mean": mean("init_bwd_win"),
            "half_open_frac": 0.0,               # no -1 sentinel convention in Argus
            "syn_rate": 0.0, "ack_rate": 0.0,     # State has no per-flag decomposition
            "fin_rate": mean("is_fin"), "rst_rate": mean("is_rst"),
            "psh_rate": 0.0, "urg_rate": 0.0,
            "proto_tcp_frac": mean("is_tcp"), "proto_udp_frac": mean("is_udp"),
            "proto_other_frac": mean("is_other"),
            "port_entropy": port_entropy, "distinct_port_frac": distinct_port_frac,
            **{f"svc_{name}_frac": v for name, v in svc_fracs.items()},
            "oneway_frac": mean("oneway"),
            "has_packet_level": 0,
            "ttl_mean": mean("sttl"), "ttl_std": std("sttl"),   # real, unlike CIC-IDS-2018
            "frag_rate": 0.0, "retrans_rate": 0.0, "payload_len_std": 0.0,
            "label_raw": dominant_label, "attack_frac": attack_frac,
            "stage": stage, "stage_idx": stage_idx, "infiltration": infiltration,
        }
        rows.append(row)

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
