"""Phase 2: task heads on top of the frozen Phase-1 world model.

Two heads now, each with its own single loss (never weighted against each
other -- they're trained in separate optimizer steps in train_heads.py).

The stage classifier was originally the only head, with infiltration
probability derived post-hoc as the softmax's mass on {Lateral Movement,
Command and Control, Exfiltration}. A head-to-head Brier/AUC comparison
against a logistic-regression baseline (trained with direct binary
supervision on exactly that target) showed this derived quantity is a real
handicap: the stage head's gradient never says "make that sum accurate," it
only says "get the 7-way classification right" -- the infiltration number
was a side-effect, never a target. InfilHead exists to close that gap: its
own linear layer, its own binary cross-entropy against the real infiltration
label, trained the same way the stage head is (real context positions +
imagined/rolled-forward positions, concatenated into one loss) -- giving it
the same direct-supervision advantage the logistic-regression baseline has.

The stage-transition matrix Pi is still not trained at all -- counted
directly from labeled sequences (Laplace-smoothed MLE), respecting day
boundaries.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attack_map import STAGES, STAGE_TO_IDX, INFILTRATION_STAGES  # noqa: E402

N_STAGES = len(STAGES)
INFIL_IDX = torch.tensor([STAGE_TO_IDX[s] for s in INFILTRATION_STAGES])


class StageHead(nn.Module):
    """Single linear layer. Kept deliberately shallow -- if this needs to be
    deep to work, the world model isn't carrying enough signal in z."""

    def __init__(self, d_latent: int, n_stages: int = N_STAGES):
        super().__init__()
        self.fc = nn.Linear(d_latent, n_stages)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)

    @staticmethod
    def infiltration_prob(stage_logits: torch.Tensor) -> torch.Tensor:
        """Kept for comparison -- this is the derived quantity that turned
        out to be poorly calibrated. InfilHead below is the direct fix."""
        p = torch.softmax(stage_logits, dim=-1)
        return p.index_select(-1, INFIL_IDX.to(p.device)).sum(-1)


class InfilHead(nn.Module):
    """Single linear layer, one output: P(infiltration). Directly
    supervised on the binary target, unlike StageHead.infiltration_prob."""

    def __init__(self, d_latent: int):
        super().__init__()
        self.fc = nn.Linear(d_latent, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z).squeeze(-1)   # logit; apply sigmoid for a probability


def fit_transition_matrix(stage_sequences: list[np.ndarray],
                          n_stages: int = N_STAGES, alpha: float = 1.0) -> np.ndarray:
    """Pi[i, j] = P(next stage = j | current stage = i), MLE + Laplace
    smoothing, counted separately per day so no transition crosses a day
    boundary (a multi-hour capture gap is not a real transition)."""
    counts = np.full((n_stages, n_stages), alpha, dtype=np.float64)
    for seq in stage_sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            counts[a, b] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def rollout_transition_matrix(pi: np.ndarray, posterior: np.ndarray, k: int) -> np.ndarray:
    """gamma_hat_{t+k} = (Pi^T)^k . gamma_t -- closed-form K-step forecast
    over the discrete stage distribution, no sampling involved."""
    pk = np.linalg.matrix_power(pi.T, k)
    return pk @ posterior
