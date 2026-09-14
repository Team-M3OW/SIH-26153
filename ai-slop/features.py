"""CIC-IDS-2018 flow CSV -> windowed, flat network-state feature matrix.

One row out = one 60-second window of the network, aggregated from however
many CICFlowMeter flow records fall in it. Streamed in chunks (some of these
CSVs are multiple GB) and aggregated with running sums, so memory use is
bounded by the number of windows in a day (~1440 at 60s), not file size.

No IP/port identity is ever used as a model *feature* -- only as an
aggregation key for the port-structure signals (entropy, well-known-service
shares, distinct-port fraction) that flag scanning behaviour. This is the
guard against the exact failure mode the network-representation literature
survey flagged: a model that quietly memorises "this IP is the attacker"
instead of learning the actual traffic pattern.

Packet-level slots (TTL, fragment flags, retransmissions, payload-size
variance) are zero-filled with an explicit `has_packet_level=0` marker --
CIC-IDS-2018's flow CSVs do not carry them. `has_packet_level` becomes 1 only
for state vectors built from a PCAP path (not implemented yet).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from attack_map import stage_for_label, STAGE_TO_IDX, INFILTRATION_STAGES

WINDOW_SECONDS = 60
CHUNK_ROWS = 200_000

# Well-known service ports worth a dedicated share feature -- chosen because
# they are exactly the services CIC-IDS-2018's own attacks target (FTP/SSH
# brute force, web attacks, SMB-adjacent lateral movement, RDP).
_SERVICE_PORTS = {
    "ftp": (20, 21), "ssh": (22,), "telnet": (23,), "smtp": (25,),
    "dns": (53,), "http": (80, 8080), "https": (443,),
    "smb": (445,), "rdp": (3389,),
}

NUMERIC_SUM_COLS = [
    "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Down/Up Ratio",
    "Init Fwd Win Byts", "Init Bwd Win Byts",
    "SYN Flag Cnt", "ACK Flag Cnt", "FIN Flag Cnt", "RST Flag Cnt",
    "PSH Flag Cnt", "URG Flag Cnt",
    "is_tcp", "is_udp", "is_other", "oneway", "half_open",
]
# Columns we also want the sum-of-squares of, to recover a std after merging
# chunks (std cannot be averaged across chunks directly).
SQ_COLS = ["Flow Duration", "Flow Byts/s", "Flow Pkts/s"]

FEATURE_COLUMNS = [
    "window_start", "n_flows",
    "duration_mean", "duration_std",
    "fwd_pkts_mean", "bwd_pkts_mean", "fwd_bytes_mean", "bwd_bytes_mean",
    "byts_per_s_mean", "byts_per_s_std", "pkts_per_s_mean", "pkts_per_s_std",
    "iat_mean_mean", "down_up_ratio_mean",
    "init_fwd_win_mean", "init_bwd_win_mean", "half_open_frac",
    "syn_rate", "ack_rate", "fin_rate", "rst_rate", "psh_rate", "urg_rate",
    "proto_tcp_frac", "proto_udp_frac", "proto_other_frac",
    "port_entropy", "distinct_port_frac",
    *[f"svc_{name}_frac" for name in _SERVICE_PORTS],
    "oneway_frac",
    "has_packet_level", "ttl_mean", "ttl_std", "frag_rate",
    "retrans_rate", "payload_len_std",
    "label_raw", "attack_frac", "stage", "stage_idx", "infiltration",
]

# The columns a model is actually allowed to see. Excludes window_start
# (metadata), label_raw/stage/stage_idx/infiltration (the targets), and
# attack_frac -- attack_frac is computed directly from the ground-truth
# label, so feeding it to the model would be leaking the answer through a
# "feature". n_flows through payload_len_std, in FEATURE_COLUMNS order.
INPUT_FEATURE_COLUMNS = [
    c for c in FEATURE_COLUMNS
    if c not in ("window_start", "label_raw", "attack_frac",
                "stage", "stage_idx", "infiltration")
]

# Heavy-tailed, non-negative count/volume/rate features whose range spans
# orders of magnitude (e.g. n_flows: benign median ~1.3K, DDoS windows up to
# 275K) -- median/IQR robust scaling alone clips anything past +-8 IQR-units
# to one identical saturated value, which erases exactly the signal that
# most clearly marks a volumetric flood. log1p first compresses that range
# without destroying the relative ordering a hard clip would. Excludes
# init_fwd_win_mean/init_bwd_win_mean (can legitimately be -1, a sentinel,
# not a count) and bounded ratios/fractions/entropy (already well-scaled).
LOG_TRANSFORM_COLUMNS = [
    "n_flows", "duration_mean", "duration_std",
    "fwd_pkts_mean", "bwd_pkts_mean", "fwd_bytes_mean", "bwd_bytes_mean",
    "byts_per_s_mean", "byts_per_s_std", "pkts_per_s_mean", "pkts_per_s_std",
    "iat_mean_mean", "syn_rate", "ack_rate", "fin_rate", "rst_rate",
    "psh_rate", "urg_rate",
]


def _clean_numeric(s: pd.Series) -> pd.Series:
    return (pd.to_numeric(s, errors="coerce")
              .replace([np.inf, -np.inf], np.nan)
              .fillna(0.0))


def _parse_timestamp(s: pd.Series) -> pd.Series:
    # CIC-IDS-2018 timestamps are "%d/%m/%Y %H:%M:%S" (24h). Fall back to
    # pandas' generic parser for the rare row that doesn't match, rather than
    # dropping the whole chunk over a handful of malformed rows.
    ts = pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    bad = ts.isna() & s.notna()
    if bad.any():
        ts.loc[bad] = pd.to_datetime(s[bad], errors="coerce")
    return ts


def _prepare_chunk(chunk: pd.DataFrame, day_start: pd.Timestamp) -> pd.DataFrame:
    # Guard against embedded duplicate-header rows, a known CIC-IDS-2018 CSV
    # quirk in some day files -- these show up as "Dst Port" being the
    # literal non-numeric string "Dst Port" rather than a header line proper.
    dst_port = pd.to_numeric(chunk["Dst Port"], errors="coerce")
    ts = _parse_timestamp(chunk["Timestamp"])
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
    """CIC-IDS-2018 flow CSV -> one row per window, flat feature vector."""
    usecols = list(dict.fromkeys(
        ["Dst Port", "Protocol", "Timestamp", *NUMERIC_SUM_COLS[:17],
         "Init Fwd Win Byts", "Init Bwd Win Byts", "TotLen Bwd Pkts", "Label"]))

    day_start = None
    sums = None          # running per-window sums, indexed by window_id
    n_flows = None        # running per-window flow counts
    port_counts: dict[int, dict[int, int]] = {}
    label_counts: dict[int, dict[str, int]] = {}

    reader = pd.read_csv(path, usecols=usecols, chunksize=chunk_rows,
                         low_memory=False)
    for raw_chunk in reader:
        if day_start is None:
            first_ts = _parse_timestamp(raw_chunk["Timestamp"]).dropna().iloc[0]
            day_start = first_ts.normalize()

        chunk = _prepare_chunk(raw_chunk, day_start)
        if chunk.empty:
            continue

        agg_cols = NUMERIC_SUM_COLS + [c + "_sq" for c in SQ_COLS]
        chunk_sums = chunk.groupby("window_id")[agg_cols].sum()
        chunk_n = chunk.groupby("window_id").size()
        sums = chunk_sums if sums is None else sums.add(chunk_sums, fill_value=0.0)
        n_flows = chunk_n if n_flows is None else n_flows.add(chunk_n, fill_value=0.0)

        port_grp = chunk.groupby(["window_id", "Dst Port"]).size()
        for (wid, port), cnt in port_grp.items():
            port_counts.setdefault(wid, {})
            port_counts[wid][port] = port_counts[wid].get(port, 0) + int(cnt)

        label_grp = chunk.groupby(["window_id", "Label"]).size()
        for (wid, label), cnt in label_grp.items():
            label_counts.setdefault(wid, {})
            label_counts[wid][label] = label_counts[wid].get(label, 0) + int(cnt)

    if sums is None:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    window_ids = sorted(sums.index.astype(int))
    rows = []
    for wid in window_ids:
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
        svc_fracs = {}
        for name, plist in _SERVICE_PORTS.items():
            svc_fracs[name] = sum(ports.get(p, 0) for p in plist) / total_port_obs

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
        stage_idx = STAGE_TO_IDX.get(stage, -1)  # -1 surfaces unmapped labels
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
