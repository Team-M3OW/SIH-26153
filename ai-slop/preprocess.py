"""Run features.process_file over CIC-IDS-2018 raw CSVs, write one output CSV
per day to hdd. Nothing is read from or written into the rejected
cyber-world-model directory's code/model/output tree -- only its raw dataset
files, which are just the public CIC-IDS-2018 download, are read from.
"""
from __future__ import annotations

import os
import sys
import time

from features import process_file

RAW_DIR = "/media/kavinder/hdd2/cyber-world-model/data/raw"
OUT_DIR = "/media/kavinder/hdd2/sih26153-processed/cic_ids2018"

DAY_FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
]


def run(files=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    files = files or DAY_FILES
    for fname in files:
        day = fname.split("_TrafficForML")[0]
        out_path = os.path.join(OUT_DIR, f"{day}.features.csv")
        in_path = os.path.join(RAW_DIR, fname)
        t0 = time.time()
        print(f"[{day}] processing {in_path} ...", flush=True)
        df = process_file(in_path)
        df.to_csv(out_path, index=False)
        n_attack = int((df["attack_frac"] > 0).sum())
        unmapped = sorted(set(df.loc[df["stage_idx"] == -1, "label_raw"]))
        print(f"[{day}] {len(df)} windows, {n_attack} with attack traffic, "
              f"stages={sorted(df['stage'].unique())}, "
              f"unmapped_labels={unmapped}, "
              f"{time.time()-t0:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    files = sys.argv[1:] or None
    run(files)
