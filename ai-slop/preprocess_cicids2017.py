"""Run features_cicids2017.process_file over all 8 CIC-IDS-2017 day files
(GeneratedLabelledFlows release -- has real timestamps, unlike the
MachineLearningCSV release), write one output CSV per day to hdd, same
schema as cic_ids2018/ and ctu13/.
"""
from __future__ import annotations

import os
import sys
import time

from features_cicids2017 import process_file

RAW_DIR = "/media/kavinder/hdd2/cyber-world-model/data/raw_cicids2017_generated"
OUT_DIR = "/media/kavinder/hdd2/sih26153-processed/cicids2017"

DAY_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


def run(files=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    files = files or DAY_FILES
    for fname in files:
        day = fname.split(".pcap_ISCX")[0]
        out_path = os.path.join(OUT_DIR, f"{day}.features.csv")
        in_path = os.path.join(RAW_DIR, fname)
        t0 = time.time()
        print(f"[{day}] processing {in_path} ...", flush=True)
        df = process_file(in_path)
        df.to_csv(out_path, index=False)
        unmapped = sorted(set(df.loc[df["stage_idx"] == -1, "label_raw"]))
        print(f"[{day}] {len(df)} windows, stages={sorted(df['stage'].unique())}, "
              f"unmapped_labels={unmapped}, {time.time()-t0:.1f}s -> {out_path}",
              flush=True)


if __name__ == "__main__":
    files = sys.argv[1:] or None
    run(files)
