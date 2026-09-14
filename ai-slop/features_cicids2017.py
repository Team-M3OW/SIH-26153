"""CIC-IDS-2017 (GeneratedLabelledFlows, raw CICFlowMeter output) -> the
same windowed feature schema as features.py, so all three flow datasets can
be trained on together.

Two adaptations from the CIC-IDS-2018 pipeline, both real, not cosmetic:

1. Column names differ slightly ("Total Fwd Packets" vs "Tot Fwd Pkts",
   "Flow Bytes/s" vs "Flow Byts/s", etc.) -- renamed to 2018's names on read
   so the exact same NUMERIC_SUM_COLS/SQ_COLS from features.py apply
   unchanged.
2. Timestamps are genuinely inconsistent across day files: Monday uses
   zero-padded "%d/%m/%Y %H:%M:%S", Tuesday/Thursday/Friday use unpadded,
   seconds-free "%-d/%-m/%Y %-H:%M". A naive pandas fallback parse without
   dayfirst=True would silently misread "4/7/2017" as April 7 instead of
   July 4 (this dataset was captured July 3-7, 2017) -- got this explicitly
   right rather than trusting the default US-first interpretation.

This release DOES carry Flow ID/Source IP/Destination IP (unlike CIC-IDS-2018's
public CSVs) -- still never used as a model feature, only implicitly
irrelevant here since we don't even read those columns in usecols.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from attack_map import stage_for_label, STAGE_TO_IDX, INFILTRATION_STAGES
from features import (FEATURE_COLUMNS, NUMERIC_SUM_COLS, SQ_COLS,
                      _SERVICE_PORTS, _clean_numeric)

WINDOW_SECONDS = 60
CHUNK_ROWS = 200_000

# 2017 column name -> 2018 column name, so the shared aggregation columns
# apply without modification.
RENAME = {
    "Destination Port": "Dst Port",
    "Total Fwd Packets": "Tot Fwd Pkts",
    "Total Backward Packets": "Tot Bwd Pkts",
    "Total Length of Fwd Packets": "TotLen Fwd Pkts",
    "Total Length of Bwd Packets": "TotLen Bwd Pkts",
    "Flow Bytes/s": "Flow Byts/s",
    "Flow Packets/s": "Flow Pkts/s",
    "Init_Win_bytes_forward": "Init Fwd Win Byts",
    "Init_Win_bytes_backward": "Init Bwd Win Byts",
    "SYN Flag Count": "SYN Flag Cnt",
    "ACK Flag Count": "ACK Flag Cnt",
    "FIN Flag Count": "FIN Flag Cnt",
    "RST Flag Count": "RST Flag Cnt",
    "PSH Flag Count": "PSH Flag Cnt",
    "URG Flag Count": "URG Flag Cnt",
}
RAW_USECOLS = ["Timestamp", "Protocol", "Label",
              "Flow Duration", "Down/Up Ratio", "Flow IAT Mean",
              *RENAME.keys()]


def _parse_timestamp_2017(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    bad = ts.isna() & s.notna()
    if bad.any():
        # dayfirst=True is load-bearing: unpadded "4/7/2017" must parse as
        # 4 July, not April 7 -- pandas' generic parser defaults US-style
        # without it.
        ts.loc[bad] = pd.to_datetime(s[bad], dayfirst=True, errors="coerce")
    return ts


def _prepare_chunk(chunk: pd.DataFrame, day_start: pd.Timestamp) -> pd.DataFrame:
    dst_port = pd.to_numeric(chunk["Dst Port"], errors="coerce")
    ts = _parse_timestamp_2017(chunk["Timestamp"])
    keep = dst_port.notna() & ts.notna()
    chunk = chunk.loc[keep].copy()
    dst_port = dst_port.loc[keep]
    ts = ts.loc[keep]

    chunk["window_id"] = ((ts - day_start).dt.total_seconds()
                          // WINDOW_SECONDS).astype(np.int64)
    chunk["Dst Port"] = dst_port.astype(np.int64)

    proto = pd.to_numeric(chunk["Protocol"], errors="coerce").fillna(-1)
    chunk["is_tcp"] = (proto == 6).astype(np.float64)
    chunk["is_udp"] = (proto == 17).astype(np.float64)
    chunk["is_other"] = (~proto.isin([6, 17])).astype(np.float64)

    bwd_bytes = _clean_numeric(chunk["TotLen Bwd Pkts"])
    chunk["oneway"] = (bwd_bytes == 0).astype(np.float64)

    init_fwd = _clean_numeric(chunk["Init Fwd Win Byts"])
    init_bwd = _clean_numeric(chunk["Init Bwd Win Byts"])
    chunk["half_open"] = ((init_fwd == -1) | (init_bwd == -1)).astype(np.float64)

    for col in NUMERIC_SUM_COLS:
        if col not in ("is_tcp", "is_udp", "is_other", "oneway", "half_open"):
            chunk[col] = _clean_numeric(chunk[col])
    for col in SQ_COLS:
        chunk[col + "_sq"] = chunk[col] ** 2
    return chunk


def process_file(path: str, window_seconds: int = WINDOW_SECONDS,
                 chunk_rows: int = CHUNK_ROWS) -> pd.DataFrame:
    day_start = None
    sums = None
    n_flows = None
    port_counts: dict[int, dict[int, int]] = {}
    label_counts: dict[int, dict[str, int]] = {}

    reader = pd.read_csv(path, usecols=RAW_USECOLS, skipinitialspace=True,
                         chunksize=chunk_rows, low_memory=False,
                         encoding="latin1")
    for raw_chunk in reader:
        raw_chunk = raw_chunk.rename(columns=RENAME)
        if day_start is None:
            first_ts = _parse_timestamp_2017(raw_chunk["Timestamp"]).dropna().iloc[0]
            day_start = first_ts.normalize()

        chunk = _prepare_chunk(raw_chunk, day_start)
        if chunk.empty:
            continue

        agg_cols = NUMERIC_SUM_COLS + [c + "_sq" for c in SQ_COLS]
        chunk_sums = chunk.groupby("window_id")[agg_cols].sum()
        chunk_n = chunk.groupby("window_id").size()
        sums = chunk_sums if sums is None else sums.add(chunk_sums, fill_value=0.0)
        n_flows = chunk_n if n_flows is None else n_flows.add(chunk_n, fill_value=0.0)

        for (wid, port), cnt in chunk.groupby(["window_id", "Dst Port"]).size().items():
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
        n_attack = sum(c for lbl, c in labels.items() if lbl.strip().lower() != "benign")
        attack_frac = n_attack / n if n > 0 else 0.0
        if n_attack > 0:
            dominant_label = max(
                ((lbl, c) for lbl, c in labels.items()
                 if lbl.strip().lower() != "benign"), key=lambda x: x[1])[0]
        else:
            dominant_label = "Benign"
        stage = stage_for_label(dominant_label)
        stage_idx = STAGE_TO_IDX.get(stage, -1)
        infiltration = int(stage in INFILTRATION_STAGES)

        row = {
            "window_start": (day_start + pd.Timedelta(
                seconds=wid * window_seconds)).isoformat(),
            "n_flows": n,
            "duration_mean": mean("Flow Duration"), "duration_std": std("Flow Duration"),
            "fwd_pkts_mean": mean("Tot Fwd Pkts"), "bwd_pkts_mean": mean("Tot Bwd Pkts"),
            "fwd_bytes_mean": mean("TotLen Fwd Pkts"), "bwd_bytes_mean": mean("TotLen Bwd Pkts"),
            "byts_per_s_mean": mean("Flow Byts/s"), "byts_per_s_std": std("Flow Byts/s"),
            "pkts_per_s_mean": mean("Flow Pkts/s"), "pkts_per_s_std": std("Flow Pkts/s"),
            "iat_mean_mean": mean("Flow IAT Mean"), "down_up_ratio_mean": mean("Down/Up Ratio"),
            "init_fwd_win_mean": mean("Init Fwd Win Byts"),
            "init_bwd_win_mean": mean("Init Bwd Win Byts"),
            "half_open_frac": mean("half_open"),
            "syn_rate": mean("SYN Flag Cnt"), "ack_rate": mean("ACK Flag Cnt"),
            "fin_rate": mean("FIN Flag Cnt"), "rst_rate": mean("RST Flag Cnt"),
            "psh_rate": mean("PSH Flag Cnt"), "urg_rate": mean("URG Flag Cnt"),
            "proto_tcp_frac": mean("is_tcp"), "proto_udp_frac": mean("is_udp"),
            "proto_other_frac": mean("is_other"),
            "port_entropy": port_entropy, "distinct_port_frac": distinct_port_frac,
            **{f"svc_{name}_frac": v for name, v in svc_fracs.items()},
            "oneway_frac": mean("oneway"),
            "has_packet_level": 0,
            "ttl_mean": 0.0, "ttl_std": 0.0, "frag_rate": 0.0,
            "retrans_rate": 0.0, "payload_len_std": 0.0,
            "label_raw": dominant_label, "attack_frac": attack_frac,
            "stage": stage, "stage_idx": stage_idx, "infiltration": infiltration,
        }
        rows.append(row)

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
