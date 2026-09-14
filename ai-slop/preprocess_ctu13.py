"""Run features_ctu13.process_file over all 13 CTU-13 scenarios, write one
output CSV per scenario to hdd, in the exact same schema as
sih26153-processed/cic_ids2018/ so the two can be loaded and trained on
together.
"""
from __future__ import annotations

import os
import sys
import time

from features_ctu13 import process_file

RAW_DIR = "/media/kavinder/hdd2/cyber-world-model/data/raw_ctu13"
OUT_DIR = "/media/kavinder/hdd2/sih26153-processed/ctu13"

SCENARIO_FILES = [
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


def run(files=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    files = files or SCENARIO_FILES
    for fname in files:
        scen = fname.split("_capture")[0]
        out_path = os.path.join(OUT_DIR, f"{scen}.features.csv")
        in_path = os.path.join(RAW_DIR, fname)
        t0 = time.time()
        print(f"[{scen}] processing {in_path} ...", flush=True)
        df = process_file(in_path)
        df.to_csv(out_path, index=False)
        unmapped = sorted(set(df.loc[df["stage_idx"] == -1, "label_raw"]))
        print(f"[{scen}] {len(df)} windows, stages={sorted(df['stage'].unique())}, "
              f"unmapped_labels={unmapped[:10]}{'...' if len(unmapped) > 10 else ''}, "
              f"{time.time()-t0:.1f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    files = sys.argv[1:] or None
    run(files)
