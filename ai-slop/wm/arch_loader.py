"""Load a Phase-1 checkpoint of either architecture (Transformer-based
WorldModel or RSSM), so Phase 2 and the comparison script can run
unmodified against whichever one was trained -- both expose the same
public interface (encode/decode/imagine/rollout), see model.py and rssm.py.
"""
from __future__ import annotations

import torch

from model import WorldModel, WorldModelConfig
from rssm import RSSM, RSSMConfig


def load_world_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    arch = ckpt.get("arch", "transformer")
    if arch.startswith("rssm"):
        cfg = RSSMConfig(**ckpt["cfg"])
        model = RSSM(cfg).to(device)
    else:
        cfg = WorldModelConfig(**ckpt["cfg"])
        model = WorldModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt, arch
