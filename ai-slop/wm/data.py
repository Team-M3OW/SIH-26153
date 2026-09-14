"""Preprocessed CIC-IDS-2018 windows -> temporally-split, scaled sequence
tensors for Phase-1 self-supervised dynamics training.

Splits are temporal (first N% of each day), never random -- adjacent
windows are strongly autocorrelated, so a random split would leak train
information into validation. Every day stays separate; a training sequence
never crosses a day boundary, since the multi-hour gap between captures is
not a transition the model should be asked to learn.

Labels (stage/infiltration) are carried through in `splits` untouched but
never fed to the model or the loss here -- Phase 1 is entirely unsupervised.
They exist so a separate evaluation script can check, post-hoc, whether
prediction error rises during known attack windows without the model ever
having been trained on that label.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import INPUT_FEATURE_COLUMNS, LOG_TRANSFORM_COLUMNS  # noqa: E402

PROCESSED_DIR = "/media/kavinder/hdd2/sih26153-processed/cic_ids2018"
PROCESSED_DIRS = [
    "/media/kavinder/hdd2/sih26153-processed/cic_ids2018",
    "/media/kavinder/hdd2/sih26153-processed/ctu13",
    # Folded in after the data-scarcity diagnosis: CTU-13 contributes only
    # Benign/C2, leaving Initial Access/Lateral Movement/Impact at 150-450
    # windows and Reconnaissance at zero. CIC-IDS-2017 directly patches all
    # four. Previously held out as a zero-shot generalization test
    # (eval_cicids2017.py) -- that test is no longer valid now that this
    # data is in the training pool; a replacement (CICIoT2023 or
    # UNSW-NB15, neither yet preprocessed) is needed if that check matters
    # again.
    "/media/kavinder/hdd2/sih26153-processed/cicids2017",
]

_LOG_MASK = np.array([c in LOG_TRANSFORM_COLUMNS for c in INPUT_FEATURE_COLUMNS])

DOMAIN_NAMES = ["cic_ids2018", "cic_ids2017", "ctu13"]


def domain_for(day_name: str) -> int:
    """Which of the three source datasets a day/scenario name belongs to --
    same classification used in compare_rollout.py's per-family breakdown,
    now also used to label windows for domain-adversarial training."""
    if day_name.startswith("scenario"):
        return DOMAIN_NAMES.index("ctu13")
    if "2018" in day_name:
        return DOMAIN_NAMES.index("cic_ids2018")
    return DOMAIN_NAMES.index("cic_ids2017")


class RobustScaler:
    """log1p on heavy-tailed count/volume columns, then median/IQR scaling,
    fit on training windows only. Without the log1p step, a hard clip at
    +-8 IQR-units flattens every DDoS-scale flow-count spike to one
    identical saturated value -- see features.LOG_TRANSFORM_COLUMNS."""

    def _log(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        X[:, _LOG_MASK] = np.log1p(np.maximum(X[:, _LOG_MASK], 0.0))
        return X

    def fit(self, X: np.ndarray) -> "RobustScaler":
        X = self._log(X)
        self.center = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        iqr = q75 - q25
        self.scale = np.where(iqr < 1e-6, 1.0, iqr)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self._log(X)
        z = (X - self.center) / self.scale
        return np.clip(z, -8.0, 8.0).astype(np.float32)


def load_days(processed_dirs: str | list[str] = PROCESSED_DIRS) -> dict[str, pd.DataFrame]:
    """Loads every *.features.csv under one or more processed directories.
    CIC-IDS-2018 day names ("Wednesday-28-02-2018") and CTU-13 scenario
    names ("scenario42") never collide, so this is a flat merge -- both
    datasets already share the exact same output schema (see
    features_ctu13.py), so nothing downstream needs to know which is which."""
    if isinstance(processed_dirs, str):
        processed_dirs = [processed_dirs]
    days = {}
    for d in processed_dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.features.csv"))):
            name = os.path.basename(path).replace(".features.csv", "")
            days[name] = pd.read_csv(path)
    return days


def temporal_split(df: pd.DataFrame, frac_train=0.70, frac_val=0.15) -> dict:
    n = len(df)
    a, b = int(n * frac_train), int(n * (frac_train + frac_val))
    return {"train": df.iloc[:a], "val": df.iloc[a:b], "test": df.iloc[b:]}


class SequenceDataset(Dataset):
    """Sliding (context+horizon) windows over one or more days' feature
    matrices. A sample never spans a day boundary (each array is one day).

    If `label_arrays` is given (one int array per day, aligned to the same
    rows as `day_arrays`), each item also returns the label sequence for the
    span -- used by Phase 2's stage head. Phase 1 leaves it as None.

    If `domain_ids` is given (one int per day -- domain doesn't change within
    a single capture), each item also returns that scalar -- used by
    domain-adversarial training in Phase 1."""

    def __init__(self, day_arrays: list[np.ndarray], context: int, horizon: int,
                label_arrays: list[np.ndarray] | None = None,
                domain_ids: list[int] | None = None):
        self.arrays = day_arrays
        self.labels = label_arrays
        self.domain_ids = domain_ids
        self.span = context + horizon
        self.index = []
        for di, arr in enumerate(day_arrays):
            for start in range(0, len(arr) - self.span + 1):
                self.index.append((di, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        di, s = self.index[i]
        seq = self.arrays[di][s: s + self.span]
        out = [torch.from_numpy(seq)]
        if self.labels is not None:
            lbl = self.labels[di][s: s + self.span]
            out.append(torch.from_numpy(lbl).long())
        if self.domain_ids is not None:
            out.append(torch.tensor(self.domain_ids[di], dtype=torch.long))
        return tuple(out) if len(out) > 1 else out[0]


def build_datasets(context: int, horizon: int, processed_dirs=PROCESSED_DIRS,
                   include_domain: bool = False):
    """Returns (datasets, scaler, splits). `splits` keeps the raw, unscaled
    per-day dataframes (with labels) for post-hoc evaluation.

    include_domain=True makes each dataset item also return which of the
    three source datasets the window came from -- opt-in, since it changes
    what a batch unpacks to (plain tensor vs. (tensor, domain_id) tuple) and
    existing callers (plain Phase-1 training, eval scripts) expect the
    plain-tensor form."""
    days = load_days(processed_dirs)
    if not days:
        raise RuntimeError(f"no preprocessed CSVs found under {processed_dirs}")
    splits = {name: temporal_split(df) for name, df in days.items()}
    names = list(splits.keys())

    train_feats = np.concatenate(
        [s["train"][INPUT_FEATURE_COLUMNS].values.astype(np.float32)
         for s in splits.values()])
    scaler = RobustScaler().fit(train_feats)

    def arrays_for(role):
        kept = [(name, s[role]) for name, s in splits.items()
               if len(s[role]) >= context + horizon]
        arrs = [scaler.transform(df[INPUT_FEATURE_COLUMNS].values.astype(np.float32))
               for _, df in kept]
        dom_ids = [domain_for(name) for name, _ in kept] if include_domain else None
        return arrs, dom_ids

    datasets = {}
    for role in ("train", "val", "test"):
        arrs, dom_ids = arrays_for(role)
        datasets[role] = SequenceDataset(arrs, context, horizon, domain_ids=dom_ids)
    return datasets, scaler, splits


def build_labeled_datasets(context: int, horizon: int, scaler: RobustScaler,
                          splits: dict, min_len: int | None = None):
    """Same splits/scaler as build_datasets, plus stage_idx per window --
    for Phase 2. Pass in the scaler and splits from the Phase-1 checkpoint's
    run so features are scaled identically; never refit the scaler here."""
    min_len = min_len or (context + horizon)

    def arrays_for(role):
        feats, labels = [], []
        for s in splits.values():
            df = s[role]
            if len(df) < min_len:
                continue
            feats.append(scaler.transform(df[INPUT_FEATURE_COLUMNS].values.astype(np.float32)))
            labels.append(df["stage_idx"].values.astype(np.int64))
        return feats, labels

    datasets = {}
    for role in ("train", "val", "test"):
        feats, labels = arrays_for(role)
        datasets[role] = SequenceDataset(feats, context, horizon, label_arrays=labels)
    return datasets


def build_labeled_datasets_with_domain(context: int, horizon: int, scaler: RobustScaler,
                                       splits: dict, min_len: int | None = None):
    """Same as build_labeled_datasets, plus a domain id per window -- for
    combining domain-adversarial training with Phase 2 (e.g. an unfrozen
    encoder fine-tuned on the classification loss AND pushed to stay
    domain-invariant at the same time, see train_heads_unfrozen_dann.py)."""
    min_len = min_len or (context + horizon)

    def arrays_for(role):
        feats, labels, domain_ids = [], [], []
        for name, s in splits.items():
            df = s[role]
            if len(df) < min_len:
                continue
            feats.append(scaler.transform(df[INPUT_FEATURE_COLUMNS].values.astype(np.float32)))
            labels.append(df["stage_idx"].values.astype(np.int64))
            domain_ids.append(domain_for(name))
        return feats, labels, domain_ids

    datasets = {}
    for role in ("train", "val", "test"):
        feats, labels, domain_ids = arrays_for(role)
        datasets[role] = SequenceDataset(feats, context, horizon, label_arrays=labels,
                                        domain_ids=domain_ids)
    return datasets
