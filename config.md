# Dataset & storage paths

All datasets and heavy artifacts live on `hdd2` — nothing is stored in this
project directory itself.

## CIC-IDS-2018 (flow-level, CICFlowMeter CSVs)

Path: `/media/kavinder/hdd2/cyber-world-model/data/raw/`

All 10 of 10 days downloaded (10.3 GB total).

| file | day / attack |
|---|---|
| Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv | FTP-BruteForce, SSH-Bruteforce |
| Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv | DoS-GoldenEye, DoS-Slowloris |
| Friday-16-02-2018_TrafficForML_CICFlowMeter.csv | DoS-SlowHTTPTest, DoS-Hulk |
| Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv | DDoS-LOIC-HTTP (note: "Thuesday" typo is in the upstream S3 filename itself) |
| Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv | DDoS-LOIC-UDP, DDoS-HOIC |
| Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv | Brute Force-Web, Brute Force-XSS, SQL Injection |
| Friday-23-02-2018_TrafficForML_CICFlowMeter.csv | same as 22-02 |
| Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv | Infiltration |
| Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv | Infiltration |
| Friday-02-03-2018_TrafficForML_CICFlowMeter.csv | Bot |

Source: `s3://cse-cic-ids2018` (AWS Open Data, `--no-sign-request`), under
`Processed Traffic Data for ML Algorithms/`.

## CTU-13 (flow-level, Argus binetflow — botnet scenarios)

Path: `/media/kavinder/hdd2/cyber-world-model/data/raw_ctu13/`

All 13 of 13 scenarios downloaded (4.6 GB total): 42, 43, 44, 45, 46, 47, 48,
49, 50, 51, 52, 53, 54 (the official CTU-13 Scenario 1–13, in their native
`CTU-Malware-Capture-Botnet-<n>` numbering — malware families span Neris,
Rbot, Virut, Menti, Sogou, Murlo, NSIS.ay).

Source: `https://mcfp.felk.cvut.cz/publicDatasets/` (Malware Capture Facility
Project mirror of `https://www.stratosphereips.org/datasets-ctu13`).

## Packet-level (PCAP) data — not downloaded

The PS allows "and/or raw PCAP files" for packet-level features. Full PCAPs
for both datasets are hundreds of GB (CIC-IDS-2018 PCAPs ~220 GB; CTU-13 full
captures are similarly large). Not fetched — flagged as an open decision, not
an oversight. Packet-only features (TTL, fragment flags, retransmissions,
payload-size distribution) are currently zeroed when only flow CSVs are used;
`cwm/pcap.py` in the prior implementation (see below) can recompute them from
real PCAPs if/when we decide to pull a subset.

## Pre-processed caches (windowed state matrices, `.npz`)

- `/media/kavinder/hdd2/cyber-world-model/data/processed/` — 60s windows
- `/media/kavinder/hdd2/cyber-world-model/data/processed_w30/` — 30s windows
- `/media/kavinder/hdd2/cyber-world-model/data/processed_s30/` — 30s stride variant

## Demo fixture

- `/media/kavinder/hdd2/cyber-world-model/data/demo_attack.pcap` — synthetic
  multi-stage PCAP (recon → brute force → lateral movement → C2), for
  exercising the packet-level path without a full PCAP download. Not used for
  any evaluation number.

## UNSW-NB15

Path: `/media/kavinder/hdd2/cyber-world-model/data/raw_unsw_nb15/`

`UNSW_NB15_training-set.csv` (31 MB, ~175K labeled records, 49 features incl.
`attack_cat`) + `NUSW-NB15_features.csv` (feature dictionary). The official
UNSW research page's download links are currently broken (site redesign);
this is the standard training-set partition via a Hugging Face mirror
(`Mouwiya/UNSW-NB15`). Only the training partition is available there — no
separate testing-set or the full 4-part raw CSVs.

## NSL-KDD (DARPA'98/99 → KDD Cup 99 successor)

Path: `/media/kavinder/hdd2/cyber-world-model/data/raw_nsl_kdd/`

`KDDTrain+.txt` + `KDDTest+.txt` (22 MB total). The original DARPA'98/99 /
KDD Cup 99 raw files at `kdd.ics.uci.edu` now return HTTP 403 outright (server
blocks direct downloads). NSL-KDD is the de-duplicated, rebalanced successor
that essentially all modern papers use when they say "the DARPA/KDD
dataset" — same 41 features and label taxonomy, without KDD99's known
redundancy/bias problems.

## MITRE ATT&CK / CAPEC / CVE-NVD (knowledge bases)

Path: `/media/kavinder/hdd2/cyber-world-model/data/raw_kb/`

- `attack-stix-enterprise.json` (52 MB) — full Enterprise ATT&CK in STIX 2.1,
  official `mitre-attack/attack-stix-data` repo.
- `capec_latest.xml` (3.7 MB) — full CAPEC attack-pattern catalog, official
  MITRE export.
- `nvd-json-data-feeds/` (3.8 GB, 386K files) — full current CVE corpus, one
  JSON record per CVE, via the community-maintained `fkie-cad/nvd-json-data-feeds`
  mirror. NVD's own bulk feed endpoints (`nvd.nist.gov/feeds/json/cve/1.1/`)
  are deprecated and return 403; NVD now only serves this via its rate-limited
  live API (services.nvd.nist.gov), which is unsuitable for an offline demo,
  hence the pre-fetched mirror.

These three are reference/enrichment knowledge bases (attack-stage mapping,
technique/pattern lookup, optional CVE-based context for flagged
ports/services) — not training data for the world model itself.

## CIC-IDS-2017

Path: `/media/kavinder/hdd2/cyber-world-model/data/raw_cicids2017/MachineLearningCSV.zip`

Registered via `cicresearch.ca`'s gated download form (name/email/org/job
title/country — submitted with user-provided credentials: Arsh Abbas,
arshabbas636@gmail.com, Delhi Technological University, Student, India) to
obtain a session token, then fetched and extracted the standard ML-ready
flow-CSV bundle (844 MB, 9 files — the 5 days split into AM/PM segments for
some days: Monday, Tuesday, Wednesday, Thursday-Morning-WebAttacks,
Thursday-Afternoon-Infiltration, Friday-Morning, Friday-Afternoon-PortScan,
Friday-Afternoon-DDoS). The raw PCAPs (`CIC-IDS-2017/PCAPs/`) were not
pulled — same hundreds-of-GB scale issue as CIC-IDS-2018's raw PCAPs.

## CICIoT2023

Path: `/media/kavinder/hdd2/cyber-world-model/data/raw_ciciot2023/MERGED_CSV.zip`

Same registration flow as CIC-IDS-2017, against `cicresearch.ca`'s IoT
dataset gate. Fetched and extracted `CSV/MERGED_CSV.zip` — the pre-merged,
paper-standard CSV version (8.7 GB, 63 `MergedNN.csv` files; 33 IoT
botnet/DDoS/recon attack classes across 105 IoT devices). The per-device
`CSV/CSV.zip` (ungrouped) and `PCAP/` raw captures were not pulled.

## LANL Comprehensive Multi-Source Cyber-Security Events ("Authentication Dataset")

Path: `/media/kavinder/hdd2/cyber-world-model/data/raw_lanl/`

Registered via `csr.lanl.gov`'s email + usage-description gate (same
credentials as above) to obtain a signed token, then fetched:

- `auth.txt.gz` (7.2 GB compressed) — the actual authentication-event log
  (58 days, Windows/AD auth events, de-identified) — this is what the PS
  means by "LANL Authentication Dataset".
- `flows.txt.gz` (1.1 GB compressed) — network flow events from internal
  routers, for cross-referencing auth events with network-level state.
- `redteam.txt.gz` (4.8 KB) — ground-truth red-team compromise events, the
  label source for lateral-movement/compromise windows.

Not fetched: `proc.txt.gz` (2.2 GB, process start/stop events) and
`dns.txt.gz` (177 MB, DNS lookups) — lower relevance to a network-flow-centric
world model; can be pulled later with the same token pattern
(`https://csr.lanl.gov/data-fence/<token>/cyber1/<file>`) if needed.

## Preprocessed feature matrices (our own pipeline, `ai-slop/`)

Path: `/media/kavinder/hdd2/sih26153-processed/cic_ids2018/`

Output of `ai-slop/preprocess.py` (which uses `ai-slop/features.py` and
`ai-slop/attack_map.py`) — one CSV per CIC-IDS-2018 day, 60-second windowed
flow aggregates (49 columns: volume/rate/timing stats, TCP flag rates,
protocol shares, port entropy + well-known-service shares + distinct-port
fraction, connection-shape signals, zero-filled packet-level placeholders,
plus `label_raw`/`stage`/`stage_idx`/`infiltration`). 3.4 MB total, all 10
days, ~55s to generate. No IP/port identity used as a feature, only as an
aggregation key. Independent of and unrelated to the `cyber-world-model`
directory's own code/pipeline — only its raw CSV *files* (the public
CIC-IDS-2018 download) are read as input.

## Prior implementation (reference, not yet adopted)

Path: `/media/kavinder/hdd2/cyber-world-model/`

A complete prior build of this same problem statement — trained world model,
MITRE ATT&CK mapping, temporal-split evaluation harness with honest reported
results, CTU-13 leave-one-source-out sweep, SHAP+attention explainability,
Streamlit app. Not under git. Provenance unconfirmed. Not yet decided whether
we build on it or treat it as reference only.
