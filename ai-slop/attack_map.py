"""Dataset attack-type labels -> MITRE ATT&CK kill-chain stages.

No dataset used in this project ships native ATT&CK labels, so this mapping
is a first-class, reviewable decision, not an implementation detail buried in
the feature pipeline. It is matched by keyword against the raw label string
(lower-cased) rather than an exact-string dict, because CIC-IDS-2018's label
casing/spacing is inconsistent across days (e.g. "DDOS attack-HOIC" vs
"DDoS attacks-LOIC-HTTP").

The PS names five stages: Reconnaissance, Initial Access, Lateral Movement,
Command & Control, Exfiltration. Two additions:

- "Impact" for volumetric DoS/DDoS: these consume resources but do not
  represent attacker progression toward compromise, so they are kept as a
  separate, explicit stage rather than force-fit into one of the five or
  silently dropped.
- "Infiltration" (CIC-IDS-2018's label for its multi-phase exploit-delivery
  scenario) is mapped to Lateral Movement: the labeled traffic in this
  dataset is the exploit-delivery-plus-internal-portscan phase, i.e. the
  attacker is already inside and moving through the network -- closer to
  Lateral Movement than to Initial Access. This is a judgment call, stated
  here rather than left implicit.
"""
from __future__ import annotations

STAGES = [
    "Benign",
    "Reconnaissance",
    "Initial Access",
    "Lateral Movement",
    "Command and Control",
    "Exfiltration",
    "Impact",
]
STAGE_TO_IDX = {s: i for i, s in enumerate(STAGES)}

# Stages that count as "the attacker is inside and progressing" -- this is
# the headline forecast target, not "any attack whatsoever".
INFILTRATION_STAGES = {"Lateral Movement", "Command and Control", "Exfiltration"}

# (keyword, stage) pairs, checked in order against the lower-cased label.
# First match wins.
_KEYWORD_RULES = [
    ("benign", "Benign"),
    ("normal", "Benign"),
    # CTU-13's "Background" traffic is unconfirmed ambient traffic -- not
    # attested as either an attack or a known-good flow. Treated as Benign
    # for training since it isn't confirmed malicious, not because it's
    # confirmed clean -- a judgment call, stated here rather than left
    # implicit in a keyword list.
    ("background", "Benign"),
    ("portscan", "Reconnaissance"),
    ("port scan", "Reconnaissance"),
    ("recon", "Reconnaissance"),
    ("bruteforce", "Initial Access"),
    ("brute force", "Initial Access"),
    ("patator", "Initial Access"),
    ("sql injection", "Initial Access"),
    ("xss", "Initial Access"),
    ("web attack", "Initial Access"),
    ("infilt", "Lateral Movement"),
    ("heartbleed", "Lateral Movement"),
    ("bot", "Command and Control"),
    ("botnet", "Command and Control"),
    ("exfil", "Exfiltration"),
    ("ddos", "Impact"),
    ("dos ", "Impact"),
    ("dos-", "Impact"),
    ("dos_", "Impact"),
    ("doattack", "Impact"),
    ("goldeneye", "Impact"),
    ("slowloris", "Impact"),
    ("slowhttptest", "Impact"),
    ("hulk", "Impact"),
]


def stage_for_label(raw_label: str) -> str:
    """Map one raw dataset label string to an ATT&CK-ish stage name.

    Falls back to "Impact" for anything containing "attack"/"dos" not
    otherwise matched, and to "Benign" only for an exact-ish benign match --
    an unrecognised non-benign label should surface as unmapped, not silently
    default to Benign and disappear from the stage distribution.
    """
    s = str(raw_label).strip().lower()
    for kw, stage in _KEYWORD_RULES:
        if kw in s:
            return stage
    if s in ("", "benign"):
        return "Benign"
    return "Unmapped:" + str(raw_label)


def stage_idx_for_label(raw_label: str) -> int:
    stage = stage_for_label(raw_label)
    if stage not in STAGE_TO_IDX:
        # Unmapped label -- surface loudly rather than defaulting silently.
        raise KeyError(
            f"Unmapped attack label: {raw_label!r} -> {stage!r}. "
            "Add a keyword rule in attack_map.py rather than defaulting."
        )
    return STAGE_TO_IDX[stage]
