"""Directly measures the train-serving distribution mismatch flagged in the
report: the live-PCAP inference path populates 5 packet-level fields
(has_packet_level, ttl_mean, ttl_std, frag_rate, retrans_rate,
payload_len_std) with genuine non-zero values computed from real packets --
values the model NEVER saw during training on CIC-IDS-2018/2017 (zero-filled
there). This takes real windows extracted from demo_attack.pcap via
live_pipeline.py, builds a zero-filled counterfactual of the SAME
underlying windows (simulating what they'd look like from a flow-CSV-only
dataset), and measures how much the model's predictions actually change --
turning "this might be a problem" into a measured number.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "wm"))
from features import INPUT_FEATURE_COLUMNS  # noqa: E402
from attack_map import STAGES  # noqa: E402
from live_pipeline import load_model, CONTEXT, K_MAX  # noqa: E402
from heads import rollout_transition_matrix, INFIL_IDX  # noqa: E402

PACKET_LEVEL_COLS = ["has_packet_level", "ttl_mean", "ttl_std", "frag_rate",
                    "retrans_rate", "payload_len_std"]
PACKET_LEVEL_IDX = [INPUT_FEATURE_COLUMNS.index(c) for c in PACKET_LEVEL_COLS]


def extract_real_windows(pcap_path: str) -> np.ndarray:
    """Reuses live_pipeline's own packet-to-window pipeline verbatim --
    these are genuinely populated packet-level fields, not synthetic."""
    from scapy.utils import PcapReader
    from scapy.layers.inet import IP
    import live_pipeline as lp

    windows: dict[int, dict] = {}
    retrans_count: dict[int, int] = {}
    day_start = [None]
    reader = PcapReader(pcap_path)
    for pkt in reader:
        t = float(pkt.time)
        ip = IP(bytes(pkt))
        lp.process_packet(ip, windows, day_start, retrans_count, t)

    rows = []
    for wid in sorted(windows.keys()):
        row = lp.finalize_window(windows[wid], retrans_count.get(wid, 0))
        if row is not None:
            rows.append(row)
    return np.stack(rows)


@torch.no_grad()
def predict(model, head, infil_head, pi, scaler, raw_windows: np.ndarray, device):
    """Runs every valid context window through the model, returns
    (stage_probs (N,7), infil_forecast (N,K_MAX))."""
    scaled = scaler.transform(raw_windows)
    stage_probs, infil_forecasts = [], []
    for t in range(CONTEXT, len(scaled)):
        ctx = torch.from_numpy(scaled[t - CONTEXT:t]).unsqueeze(0).to(device)
        z = model.encode(ctx)
        gamma_t = torch.softmax(head(z[:, -1]), dim=-1).squeeze(0).cpu().numpy()
        out = model.rollout(ctx, k=K_MAX, n_traj=64)
        lat = out["latents"].squeeze(0)
        infil = torch.sigmoid(infil_head(lat)).mean(dim=1).cpu().numpy()
        stage_probs.append(gamma_t)
        infil_forecasts.append(infil)
    return np.stack(stage_probs), np.stack(infil_forecasts)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, head, infil_head, pi, scaler = load_model(device)

    pcap_path = "/media/kavinder/hdd2/cyber-world-model/data/demo_attack.pcap"
    print(f"extracting real packet-level windows from {pcap_path} ...")
    real_windows = extract_real_windows(pcap_path)
    print(f"  {len(real_windows)} windows, packet-level columns real (non-zero where applicable)")

    zero_filled = real_windows.copy()
    zero_filled[:, PACKET_LEVEL_IDX] = 0.0
    print(f"  built zero-filled counterfactual (simulating a flow-CSV-only source dataset)")

    print("\nrunning both through the model...")
    stage_real, infil_real = predict(model, head, infil_head, pi, scaler, real_windows, device)
    stage_zero, infil_zero = predict(model, head, infil_head, pi, scaler, zero_filled, device)

    pred_stage_real = stage_real.argmax(-1)
    pred_stage_zero = stage_zero.argmax(-1)
    agree = (pred_stage_real == pred_stage_zero).mean()

    kl = (stage_real * (np.log(stage_real + 1e-9) - np.log(stage_zero + 1e-9))).sum(-1)
    infil_diff = np.abs(infil_real - infil_zero)

    print(f"\nn windows evaluated: {len(pred_stage_real)}")
    print(f"stage prediction agreement (real-packet vs zero-filled): {agree*100:.1f}%")
    print(f"stage distribution KL divergence (real->zero), mean={kl.mean():.4f}  max={kl.max():.4f}")
    print(f"infiltration forecast |difference|, per horizon k=1..{K_MAX}: "
         f"mean={infil_diff.mean(axis=0)}  max={infil_diff.max(axis=0)}")

    disagreements = np.where(pred_stage_real != pred_stage_zero)[0]
    if len(disagreements):
        print(f"\n{len(disagreements)} windows where the stage call FLIPS depending on whether "
             f"packet-level fields are real or zero-filled:")
        for i in disagreements[:5]:
            print(f"  window {i}: real-packet pred={STAGES[pred_stage_real[i]]}  "
                 f"zero-filled pred={STAGES[pred_stage_zero[i]]}")


if __name__ == "__main__":
    main()
