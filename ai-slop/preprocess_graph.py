"""Build per-host communication graphs (see graph_features.py) for CTU-13
and CIC-IDS-2017 -- the two datasets that carry real host IPs. CIC-IDS-2018
is not covered (its public CSVs have no IP columns). Output: one .npz per
scenario/day, matching the existing flat pipeline's naming convention
(scenario name via `fname.split("_capture")[0]`, day name via
`fname.split(".pcap_ISCX")[0]`) so graph windows can be joined to the
existing labeled flat CSVs by window_start.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

from graph_features import build_graph_windows

CTU13_RAW_DIR = "/media/kavinder/hdd2/cyber-world-model/data/raw_ctu13"
CTU13_OUT_DIR = "/media/kavinder/hdd2/sih26153-processed/graph_ctu13"
CIC2017_RAW_DIR = "/media/kavinder/hdd2/cyber-world-model/data/raw_cicids2017_generated"
CIC2017_OUT_DIR = "/media/kavinder/hdd2/sih26153-processed/graph_cicids2017"

CTU13_SCENARIO_FILES = [
    "scenario42_capture20110810.binetflow.csv",
    "scenario43_capture20110811.binetflow.csv",
    "scenario44_capture20110812.binetflow.csv",
    "scenario45_capture20110815.binetflow.csv",
    "scenario46_capture20110815-2.binetflow.csv",
    "scenario47_capture20110816.binetflow.csv",
    "scenario48_capture20110816-2.binetflow.csv",
    "scenario49_capture20110816-3.binetflow.csv",
    "scenario50_capture20110817.binetflow.csv",
    "scenario51_capture20110818.binetflow.csv",
    "scenario52_capture20110818-2.binetflow.csv",
    "scenario53_capture20110819.binetflow.csv",
    "scenario54_capture20110815-3.binetflow.csv",
]
CIC2017_DAY_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


def _clean(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def process_ctu13(path: str, chunk_rows: int = 500_000, nrows: int | None = None) -> dict:
    usecols = ["SrcAddr", "DstAddr", "Proto", "Dport", "State", "StartTime",
              "SrcPkts", "DstPkts", "SrcBytes", "DstBytes", "Dur"]
    day_start = None
    pieces = []
    reader = pd.read_csv(path, usecols=usecols, chunksize=chunk_rows,
                         low_memory=False, nrows=nrows)
    for chunk in reader:
        ts = pd.to_datetime(chunk["StartTime"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
        bad = ts.isna() & chunk["StartTime"].notna()
        if bad.any():
            ts.loc[bad] = pd.to_datetime(chunk["StartTime"][bad], errors="coerce")
        if day_start is None and ts.notna().any():
            # Matches features_ctu13.process_file's t0 exactly: first valid
            # TIMESTAMP in the chunk, before any Dport filtering -- getting
            # this wrong shifts every window boundary and breaks the join
            # to the existing labeled flat CSVs by window_start.
            day_start = ts.dropna().iloc[0]
        dport = pd.to_numeric(chunk["Dport"], errors="coerce")
        keep = ts.notna() & dport.notna()
        if not keep.any():
            continue
        c = chunk.loc[keep]
        ts, dport = ts.loc[keep], dport.loc[keep]

        proto = c["Proto"].astype(str).str.lower()
        state = c["State"].astype(str)
        pieces.append(pd.DataFrame({
            "timestamp": ts, "src": c["SrcAddr"], "dst": c["DstAddr"],
            "dport": dport.astype(np.int64), "is_tcp": (proto == "tcp").astype(np.float32),
            "dur": _clean(c["Dur"]).astype(np.float32),
            "src_bytes": _clean(c["SrcBytes"]).astype(np.float32),
            "dst_bytes": _clean(c["DstBytes"]).astype(np.float32),
            "src_pkts": _clean(c["SrcPkts"]).astype(np.float32),
            "syn": state.str.contains("S", case=False, na=False).astype(np.float32),
        }))
    if not pieces:
        return None
    df = pd.concat(pieces, ignore_index=True)
    return build_graph_windows(df, day_start)


def process_cicids2017(path: str, chunk_rows: int = 200_000, nrows: int | None = None) -> dict:
    usecols = ["Timestamp", "Protocol", "Source IP", "Destination IP", "Destination Port",
              "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
              "Total Length of Fwd Packets", "Total Length of Bwd Packets", "SYN Flag Count"]
    day_start = None
    pieces = []
    reader = pd.read_csv(path, usecols=usecols, skipinitialspace=True, chunksize=chunk_rows,
                         low_memory=False, encoding="latin1", nrows=nrows)
    for chunk in reader:
        ts = pd.to_datetime(chunk["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        bad = ts.isna() & chunk["Timestamp"].notna()
        if bad.any():
            ts.loc[bad] = pd.to_datetime(chunk["Timestamp"][bad], dayfirst=True, errors="coerce")
        if day_start is None and ts.notna().any():
            # Matches features_cicids2017.process_file's day_start exactly
            # -- CIC-2017's flat pipeline normalizes to midnight (CTU-13's
            # does not) -- getting this wrong shifts every window boundary.
            day_start = ts.dropna().iloc[0].normalize()
        dst_port = pd.to_numeric(chunk["Destination Port"], errors="coerce")
        keep = ts.notna() & dst_port.notna()
        if not keep.any():
            continue
        c = chunk.loc[keep]
        ts, dst_port = ts.loc[keep], dst_port.loc[keep]

        proto = pd.to_numeric(c["Protocol"], errors="coerce").fillna(-1)
        syn_cnt = _clean(c["SYN Flag Count"])
        pieces.append(pd.DataFrame({
            "timestamp": ts, "src": c["Source IP"], "dst": c["Destination IP"],
            "dport": dst_port.astype(np.int64), "is_tcp": (proto == 6).astype(np.float32),
            "dur": (_clean(c["Flow Duration"]) / 1e6).astype(np.float32),  # us -> s
            "src_bytes": _clean(c["Total Length of Fwd Packets"]).astype(np.float32),
            "dst_bytes": _clean(c["Total Length of Bwd Packets"]).astype(np.float32),
            "src_pkts": _clean(c["Total Fwd Packets"]).astype(np.float32),
            "syn": (syn_cnt > 0).astype(np.float32),
        }))
    if not pieces:
        return None
    df = pd.concat(pieces, ignore_index=True)
    return build_graph_windows(df, day_start)


def _save(out: dict, out_path: str):
    np.savez_compressed(out_path, window_id=out["window_id"],
                        node_feats=out["node_feats"], adjacency=out["adjacency"],
                        mask=out["mask"])


def run_ctu13(files=None, nrows=None):
    os.makedirs(CTU13_OUT_DIR, exist_ok=True)
    for fname in (files or CTU13_SCENARIO_FILES):
        scen = fname.split("_capture")[0]
        out_path = os.path.join(CTU13_OUT_DIR, f"{scen}.graph.npz")
        t0 = time.time()
        print(f"[{scen}] processing ...", flush=True)
        out = process_ctu13(os.path.join(CTU13_RAW_DIR, fname), nrows=nrows)
        if out is None:
            print(f"[{scen}] no valid rows, skipped"); continue
        _save(out, out_path)
        print(f"[{scen}] {len(out['window_id'])} windows, "
              f"{time.time()-t0:.1f}s -> {out_path}", flush=True)


def run_cicids2017(files=None, nrows=None):
    os.makedirs(CIC2017_OUT_DIR, exist_ok=True)
    for fname in (files or CIC2017_DAY_FILES):
        day = fname.split(".pcap_ISCX")[0]
        out_path = os.path.join(CIC2017_OUT_DIR, f"{day}.graph.npz")
        t0 = time.time()
        print(f"[{day}] processing ...", flush=True)
        out = process_cicids2017(os.path.join(CIC2017_RAW_DIR, fname), nrows=nrows)
        if out is None:
            print(f"[{day}] no valid rows, skipped"); continue
        _save(out, out_path)
        print(f"[{day}] {len(out['window_id'])} windows, "
              f"{time.time()-t0:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "ctu13"):
        run_ctu13()
    if which in ("all", "cicids2017"):
        run_cicids2017()
