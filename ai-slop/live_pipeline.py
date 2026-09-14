"""Live/replay flow-feature extraction + forecasting + SHAP explanation.

Two data sources, same downstream pipeline:
  --source pcap  : replay demo_attack.pcap (or any pcap) at real packet
                   timing -- no root needed, no privacy concern (synthetic
                   traffic). Default, safe to run immediately.
  --source live  : sniff a real interface with scapy. Needs root (raw
                   sockets) -- run this script with sudo yourself; it is
                   never elevated automatically. Captures real traffic from
                   whatever network this machine is on -- only use this on
                   a network you're authorized to monitor.

Builds the SAME 43-feature schema as features.py, computed from raw
packets instead of pre-aggregated flow CSVs -- which means ttl_mean/std,
frag_rate, and payload_len_std are REAL here (features.py zero-fills them
for CIC-IDS-2018/2017's flow-CSV-only pipeline; this live path has actual
packets to compute them from). Every 60s of packet time, a window closes,
gets scaled with the training run's own RobustScaler, appended to a
rolling context buffer, and once >=16 windows exist, run through the
chosen checkpoint (RSSM, encoder unfrozen) for:
  - K=1..5 step infiltration probability forecast
  - current-window MITRE stage classification
  - a SHAP explanation of which of the 43 features drove that stage call
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "wm"))
from features import FEATURE_COLUMNS, INPUT_FEATURE_COLUMNS, _SERVICE_PORTS  # noqa: E402
from data import RobustScaler  # noqa: E402
from heads import StageHead, InfilHead, rollout_transition_matrix, INFIL_IDX  # noqa: E402
from attack_map import STAGES  # noqa: E402
from arch_loader import load_world_model  # noqa: E402

WINDOW_SECONDS = 60
CONTEXT = 16
K_MAX = 5
CKPT_DIR = "/media/kavinder/hdd2/sih26153-processed/checkpoints"


# ---------------------------------------------------------------- flows ---

class FlowAcc:
    __slots__ = ("fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes",
                "syn", "ack", "fin", "rst", "psh", "urg",
                "ttls", "frag", "payload_lens", "init_fwd_win", "init_bwd_win",
                "t0", "t1", "dport", "is_tcp", "is_udp", "fwd_src", "seen_seq")

    def __init__(self, fwd_src, dport, is_tcp, is_udp, t):
        self.fwd_pkts = self.bwd_pkts = 0
        self.fwd_bytes = self.bwd_bytes = 0
        self.syn = self.ack = self.fin = self.rst = self.psh = self.urg = 0
        self.ttls = []
        self.frag = 0
        self.payload_lens = []
        self.init_fwd_win = self.init_bwd_win = -1
        self.t0 = self.t1 = t
        self.dport = dport
        self.is_tcp, self.is_udp = is_tcp, is_udp
        self.fwd_src = fwd_src
        self.seen_seq = defaultdict(set)   # direction -> set of seq numbers, for a crude retrans count


def _flow_key(ip):
    proto = ip.proto
    sport = getattr(ip.payload, "sport", 0)
    dport = getattr(ip.payload, "dport", 0)
    a, b = (ip.src, sport), (ip.dst, dport)
    # unordered key so both directions of the same flow collide
    if a <= b:
        return (a, b, proto), True   # True: this packet is in the "canonical fwd" direction
    return (b, a, proto), False


def process_packet(ip, windows: dict, day_start: list, retrans_count: dict, t: float):
    """t: the packet's REAL capture timestamp -- must come from the
    original packet object read off the wire/pcap, not `ip.time` after
    `IP(bytes(pkt))` reconstruction (that resets .time to object-
    construction wall-clock time, silently, since the new IP object never
    inherits the original frame's capture timestamp)."""
    if day_start[0] is None:
        day_start[0] = t
    wid = int((t - day_start[0]) // WINDOW_SECONDS)
    key, is_fwd = _flow_key(ip)
    flows = windows.setdefault(wid, {})
    f = flows.get(key)
    is_tcp, is_udp = ip.proto == 6, ip.proto == 17
    dport = getattr(ip.payload, "dport", 0) if (is_tcp or is_udp) else 0
    if f is None:
        f = FlowAcc(fwd_src=ip.src if is_fwd else ip.dst, dport=dport,
                   is_tcp=is_tcp, is_udp=is_udp, t=t)
        flows[key] = f

    length = len(ip)
    payload_len = len(bytes(ip.payload.payload)) if hasattr(ip.payload, "payload") else 0
    fragmented = (int(ip.flags) & 0x1) or (ip.frag > 0)   # MF bit or nonzero offset
    f.ttls.append(ip.ttl)
    f.payload_lens.append(payload_len)
    if fragmented:
        f.frag += 1
    f.t1 = t

    fwd = (ip.src == f.fwd_src)
    if fwd:
        f.fwd_pkts += 1; f.fwd_bytes += length
    else:
        f.bwd_pkts += 1; f.bwd_bytes += length

    if is_tcp:
        flags = ip.payload.flags
        if flags & 0x02: f.syn += 1
        if flags & 0x10: f.ack += 1
        if flags & 0x01: f.fin += 1
        if flags & 0x04: f.rst += 1
        if flags & 0x08: f.psh += 1
        if flags & 0x20: f.urg += 1
        win = ip.payload.window
        if fwd and f.init_fwd_win == -1: f.init_fwd_win = win
        if not fwd and f.init_bwd_win == -1: f.init_bwd_win = win
        seq = ip.payload.seq
        direction = "fwd" if fwd else "bwd"
        if seq in f.seen_seq[direction]:
            retrans_count[wid] = retrans_count.get(wid, 0) + 1
        f.seen_seq[direction].add(seq)
    return wid


def finalize_window(flows: dict, n_retrans: int) -> np.ndarray:
    n = len(flows)
    if n == 0:
        return None

    def stat(vals):
        vals = np.array(vals, dtype=np.float64)
        return float(vals.mean()) if len(vals) else 0.0, float(vals.std()) if len(vals) else 0.0

    durations, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes = [], [], [], [], []
    byts_per_s, pkts_per_s, down_up = [], [], []
    init_fwd_wins, init_bwd_wins, half_opens = [], [], []
    syns, acks, fins, rsts, pshs, urgs = [], [], [], [], [], []
    is_tcps, is_udps, oneways = [], [], []
    ttl_means, frag_flags, payload_stds = [], [], []
    port_counts: dict[int, int] = {}
    all_payload_lens = []

    for f in flows.values():
        dur = max(f.t1 - f.t0, 1e-6) * 1e6   # microseconds, matching CICFlowMeter units
        durations.append(dur)
        fwd_pkts.append(f.fwd_pkts); bwd_pkts.append(f.bwd_pkts)
        fwd_bytes.append(f.fwd_bytes); bwd_bytes.append(f.bwd_bytes)
        tot_bytes, tot_pkts = f.fwd_bytes + f.bwd_bytes, f.fwd_pkts + f.bwd_pkts
        byts_per_s.append(tot_bytes / (dur / 1e6) if dur > 0 else 0.0)
        pkts_per_s.append(tot_pkts / (dur / 1e6) if dur > 0 else 0.0)
        down_up.append(f.bwd_bytes / f.fwd_bytes if f.fwd_bytes > 0 else 0.0)
        init_fwd_wins.append(f.init_fwd_win); init_bwd_wins.append(f.init_bwd_win)
        half_opens.append(1.0 if (f.init_fwd_win == -1 or f.init_bwd_win == -1) else 0.0)
        syns.append(f.syn); acks.append(f.ack); fins.append(f.fin)
        rsts.append(f.rst); pshs.append(f.psh); urgs.append(f.urg)
        is_tcps.append(1.0 if f.is_tcp else 0.0); is_udps.append(1.0 if f.is_udp else 0.0)
        oneways.append(1.0 if f.bwd_bytes == 0 else 0.0)
        ttl_means.append(np.mean(f.ttls) if f.ttls else 0.0)
        frag_flags.append(1.0 if f.frag > 0 else 0.0)
        payload_stds.append(np.std(f.payload_lens) if f.payload_lens else 0.0)
        all_payload_lens.extend(f.payload_lens)
        port_counts[f.dport] = port_counts.get(f.dport, 0) + 1

    total_port_obs = sum(port_counts.values()) or 1
    probs = np.array([c / total_port_obs for c in port_counts.values()])
    port_entropy = float(-(probs * np.log2(probs)).sum()) if len(probs) else 0.0
    distinct_port_frac = len(port_counts) / n
    svc_fracs = {name: sum(port_counts.get(p, 0) for p in plist) / total_port_obs
                for name, plist in _SERVICE_PORTS.items()}

    row = {
        "n_flows": n,
        "duration_mean": np.mean(durations), "duration_std": np.std(durations),
        "fwd_pkts_mean": np.mean(fwd_pkts), "bwd_pkts_mean": np.mean(bwd_pkts),
        "fwd_bytes_mean": np.mean(fwd_bytes), "bwd_bytes_mean": np.mean(bwd_bytes),
        "byts_per_s_mean": np.mean(byts_per_s), "byts_per_s_std": np.std(byts_per_s),
        "pkts_per_s_mean": np.mean(pkts_per_s), "pkts_per_s_std": np.std(pkts_per_s),
        "iat_mean_mean": 0.0, "down_up_ratio_mean": np.mean(down_up),
        "init_fwd_win_mean": np.mean(init_fwd_wins), "init_bwd_win_mean": np.mean(init_bwd_wins),
        "half_open_frac": np.mean(half_opens),
        "syn_rate": np.mean(syns), "ack_rate": np.mean(acks), "fin_rate": np.mean(fins),
        "rst_rate": np.mean(rsts), "psh_rate": np.mean(pshs), "urg_rate": np.mean(urgs),
        "proto_tcp_frac": np.mean(is_tcps), "proto_udp_frac": np.mean(is_udps),
        "proto_other_frac": 1.0 - np.mean(is_tcps) - np.mean(is_udps),
        "port_entropy": port_entropy, "distinct_port_frac": distinct_port_frac,
        **{f"svc_{name}_frac": v for name, v in svc_fracs.items()},
        "oneway_frac": np.mean(oneways),
        "has_packet_level": 1,   # real packet data -- unlike the flat CIC pipeline
        "ttl_mean": np.mean(ttl_means), "ttl_std": np.std(ttl_means),
        "frag_rate": np.mean(frag_flags),
        "retrans_rate": n_retrans / max(sum(fwd_pkts) + sum(bwd_pkts), 1),
        "payload_len_std": np.std(all_payload_lens) if all_payload_lens else 0.0,
    }
    return np.array([row[c] for c in INPUT_FEATURE_COLUMNS], dtype=np.float32)


# ---------------------------------------------------------------- model ---

def load_model(device, phase1_ckpt=None, phase2_ckpt=None):
    phase1_ckpt = phase1_ckpt or os.path.join(CKPT_DIR, "rssm_unfrozen_phase1.pt")
    phase2_ckpt = phase2_ckpt or os.path.join(CKPT_DIR, "rssm_unfrozen_phase2_head.pt")
    model, cfg, ckpt1, arch = load_world_model(phase1_ckpt, device)
    ckpt2 = torch.load(phase2_ckpt, map_location=device)
    head = StageHead(cfg.d_latent).to(device); head.load_state_dict(ckpt2["head"]); head.eval()
    infil_head = InfilHead(cfg.d_latent).to(device); infil_head.load_state_dict(ckpt2["infil_head"]); infil_head.eval()
    pi = ckpt2["pi"].cpu().numpy()
    scaler = RobustScaler()
    scaler.center = ckpt1["scaler_center"].cpu().numpy()
    scaler.scale = ckpt1["scaler_scale"].cpu().numpy()
    return model, head, infil_head, pi, scaler


def explain_current_window(model, head, scaler, context_raw: np.ndarray, device):
    """SHAP KernelExplainer over the 43 raw features of the CURRENT (last)
    window, holding the rest of the context fixed -- explains "given the
    prior 15 minutes, which of THIS window's features drove the stage
    call." Background = a modest random sample of plausible feature values
    (Gaussian around the current window, in raw units) since we don't
    carry the training set along at inference time."""
    import shap

    ctx_scaled_prefix = scaler.transform(context_raw[:-1])   # (context-1, F), fixed
    last_raw = context_raw[-1]

    def predict(X_raw: np.ndarray) -> np.ndarray:
        X_scaled = scaler.transform(X_raw.astype(np.float32))
        seqs = np.stack([np.concatenate([ctx_scaled_prefix, x[None, :]], axis=0) for x in X_scaled])
        with torch.no_grad():
            t = torch.from_numpy(seqs).to(device)
            z = model.encode(t)
            probs = torch.softmax(head(z[:, -1]), dim=-1).cpu().numpy()
        return probs

    background = last_raw[None, :] + np.random.randn(30, len(last_raw)) * (np.abs(last_raw) * 0.1 + 1e-3)
    explainer = shap.KernelExplainer(predict, background)
    shap_values = explainer.shap_values(last_raw[None, :], nsamples=100, silent=True)
    pred_stage = int(np.argmax(predict(last_raw[None, :])[0]))
    sv = shap_values[0][:, pred_stage] if isinstance(shap_values, list) else shap_values[..., pred_stage][0]
    order = np.argsort(-np.abs(sv))[:8]
    return STAGES[pred_stage], [(INPUT_FEATURE_COLUMNS[i], float(sv[i]), float(last_raw[i])) for i in order]


# --------------------------------------------------------------- driver ---

def run(source: str, pcap_path: str | None, iface: str | None, speed: float,
       phase1_ckpt=None, phase2_ckpt=None):
    from scapy.layers.inet import IP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, head, infil_head, pi, scaler = load_model(device, phase1_ckpt, phase2_ckpt)
    print(f"model loaded on {device}\n")

    windows: dict[int, dict] = {}
    retrans_count: dict[int, int] = {}
    day_start = [None]
    context_buf: list[np.ndarray] = []
    closed_upto = [-1]

    def process_new_window(row: np.ndarray, label: str = "", true_stage: str | None = None):
        context_buf.append(row)
        if len(context_buf) > CONTEXT:
            context_buf.pop(0)
        print(f"\n--- window{label} closed, {len(context_buf)}/{CONTEXT} in context ---"
             + (f"  [ground truth: {true_stage}]" if true_stage is not None else ""))
        if len(context_buf) < CONTEXT:
            return

        ctx_raw = np.stack(context_buf)
        ctx_scaled = scaler.transform(ctx_raw)
        with torch.no_grad():
            t = torch.from_numpy(ctx_scaled).unsqueeze(0).to(device)
            z = model.encode(t)
            gamma_t = torch.softmax(head(z[:, -1]), dim=-1).squeeze(0).cpu().numpy()
            cur_stage = STAGES[int(gamma_t.argmax())]
            out = model.rollout(t, k=K_MAX, n_traj=64)
            lat = out["latents"].squeeze(0)
            infil_neural = torch.sigmoid(infil_head(lat)).mean(dim=1).cpu().numpy()
        infil_pi = np.array([rollout_transition_matrix(pi, gamma_t, k)[INFIL_IDX.numpy()].sum()
                             for k in range(1, K_MAX + 1)])

        print(f"current stage: {cur_stage}")
        print(f"infiltration probability forecast (k=1..{K_MAX}): "
              f"{[f'{p:.3f}' for p in infil_neural]}  (Pi check: {[f'{p:.3f}' for p in infil_pi]})")

        try:
            stage_name, top_features = explain_current_window(model, head, scaler, ctx_raw, device)
            print(f"SHAP explanation for '{stage_name}' classification (top features, raw value):")
            for name, val, raw in top_features:
                print(f"    {name:20s} shap={val:+.4f}  raw_value={raw:.3f}")
        except Exception as e:
            print(f"  (SHAP explanation failed: {e})")

    def on_window_closed(wid: int):
        row = finalize_window(windows.pop(wid, {}), retrans_count.pop(wid, 0))
        if row is not None:
            process_new_window(row, label=f" {wid}")

    def feed(ip, t: float):
        wid = process_packet(ip, windows, day_start, retrans_count, t)
        while closed_upto[0] < wid - 1:
            closed_upto[0] += 1
            if closed_upto[0] in windows or closed_upto[0] < wid - 1:
                on_window_closed(closed_upto[0])

    if source == "csv":
        # Real attack day, already-extracted flow-level features (see
        # features.py) -- no packet parsing, since there's no PCAP to
        # parse (raw CIC-IDS-2018 PCAPs were never downloaded, ~220GB).
        # This validates the SAME downstream inference+SHAP pipeline
        # against genuine attack data, just skipping the packet-ingestion
        # step this file also builds. Ground truth stage is printed
        # alongside the prediction for direct comparison -- note this day
        # WAS part of Phase 1/2 training's pooled dataset, so this is a
        # pipeline-correctness demonstration, not a generalization test
        # (see the dedicated cross-episode test for that).
        import pandas as pd
        path = pcap_path or "/media/kavinder/hdd2/sih26153-processed/cic_ids2018/Wednesday-28-02-2018.features.csv"
        print(f"replaying real attack day: {path} ...")
        df = pd.read_csv(path)
        for i, r in df.iterrows():
            row = r[INPUT_FEATURE_COLUMNS].values.astype(np.float32)
            process_new_window(row, label=f" {i}", true_stage=r["stage"])
            if speed > 0:
                time.sleep(1.0 / speed)
        return

    if source == "pcap":
        from scapy.utils import PcapReader
        path = pcap_path or "/media/kavinder/hdd2/cyber-world-model/data/demo_attack.pcap"
        print(f"replaying {path} ...")
        reader = PcapReader(path)
        last_t = None
        for pkt in reader:
            t = float(pkt.time)          # capture BEFORE reconstruction -- see process_packet's docstring
            ip = IP(bytes(pkt))
            if last_t is not None and speed > 0:
                time.sleep(max(0.0, (t - last_t) / speed))
            last_t = t
            feed(ip, t)
        for wid in sorted(windows.keys()):
            on_window_closed(wid)
    elif source == "live":
        from scapy.all import sniff

        def on_pkt(pkt):
            t = float(pkt.time)
            if pkt.haslayer(IP):
                feed(pkt[IP], t)
        print(f"sniffing {iface} live -- Ctrl+C to stop ...")
        sniff(iface=iface, prn=on_pkt, store=False)
    else:
        raise ValueError(source)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["pcap", "live", "csv"], default="pcap")
    p.add_argument("--pcap", default=None,
                  help="path to a pcap file (source=pcap) or a features.csv (source=csv)")
    p.add_argument("--iface", default=None, help="interface to sniff (source=live, needs sudo)")
    p.add_argument("--speed", type=float, default=200.0,
                  help="replay speed multiplier for source=pcap (200x default so a 45min capture "
                       "runs in ~13s); for source=csv, rows/second (0 = as fast as possible)")
    args = p.parse_args()
    run(args.source, args.pcap, args.iface, args.speed)
