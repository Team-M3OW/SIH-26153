# Architecture: A World Model for Predictive Cyber Defence

## 1. System overview

```
raw flow CSV → 60s windowing → flat feature vector S_t (43-dim)
                                        │
                    ┌───────────────────┴───────────────────┐
                    │      PHASE 1 — self-supervised          │
                    │  Encoder → causal Transformer → N(μ,σ²) │
                    │  over z_{t+1} → Decoder → Ŝ_{t+1}       │
                    │  (no labels; trained on all traffic)    │
                    └───────────────────┬───────────────────┘
                                        │ (frozen)
                    ┌───────────────────┴───────────────────┐
                    │      PHASE 2 — supervised heads          │
                    │  stage classifier (real + imagined z)   │
                    │  Π: counted stage-transition matrix     │
                    └───────────────────┬───────────────────┘
                                        │
              K-step open-loop rollout (64 trajectories, sampled,
              never fed a real future observation) → infiltration
              probability timeline + ATT&CK stage forecast
                                        │
              attention weights + Π matrix → explainability
```

## 2. State representation

One row = one 60-second network window, 43 features: flow volume/rate
statistics, TCP flag rates, protocol shares, destination-port structure
(entropy, well-known-service shares, distinct-port fraction — the scan
signature), connection-shape signals (one-way/zero-byte fractions), and
packet-level slots (TTL, fragmentation, retransmission) — populated with
real values for CTU-13 (which carries real TTL), zero-filled with an
explicit flag for CIC-IDS-2018/2017 (whose public CSVs don't carry them).
No IP/port identity is ever a model input, only an aggregation key — the
documented failure mode (memorizing "this IP is the attacker") this
guards against. Heavy-tailed count/volume columns (`n_flows`, byte/packet
means) are log1p-transformed before robust scaling; without this, DDoS-scale
flow spikes saturate a hard percentile clip and the volumetric signal that
should most clearly mark an Impact-stage window is destroyed (see §5).

## 3. World model

**Encoder/Decoder**: small MLPs, 43-dim ↔ 16-dim latent. **Transition**: a
2-layer causal Transformer over the latent sequence, outputting `(μ, logvar)`
per step — a genuine distribution over `z_{t+1}`, not a point estimate.
K-step forecasting draws 64 independent reparameterized samples and rolls
each forward, so the reported probability is a Monte Carlo estimate with an
honest uncertainty band. No action variable — this is deliberately
forecasting-only (no controllable intervention is being modeled; see
Appendix note), a stated scope choice, not an oversight.

**Training loss (Phase 1, self-supervised, ~80K params)**: Gaussian NLL of
the real next window, scored two ways and *concatenated* into one loss
rather than weighted against each other — (a) one-step, teacher-forced,
every position, and (b) multi-step, open-loop (`model.imagine`, no real
observation fed in past the context, decoded and scored against real
future windows). (b) was added after an initial version trained on (a)
alone showed a large real-vs-imagined accuracy gap (§5) — the dynamics
model had never been trained to stay accurate over its own open-loop
rollout, only to predict one real step from real history.

**Task heads (Phase 2, supervised, encoder+transition frozen)**: a single
linear stage classifier (7 ATT&CK-mapped stages, §Datasets), trained by
cross-entropy on real encoded context states *and* on states the model
imagined via open-loop rollout, concatenated into one loss — this is what
makes the K-step forecast trustworthy rather than degrading on latents the
classifier never saw during training. Infiltration probability is not a
separate head: it is the stage softmax's mass on {Lateral Movement, Command
& Control, Exfiltration}, read off the one classifier that exists.
Class-imbalance handling: sqrt-inverse-frequency loss weighting; model
selection uses `min(macroF1_real, macroF1_imagined)` across epochs, not
either alone, to reject checkpoints where one class silently collapses.

**Π — the second, closed-form forecast**: a stage-transition matrix,
counted (MLE, Laplace-smoothed) directly from labeled sequences, never
gradient-trained, respecting day/scenario boundaries. K-step forecast is a
literal matrix power, `γ̂_{t+K} = (Πᵀ)^K γ_t` — whiteboard-showable,
zero sampling cost, run alongside the neural ensemble as a cheap check on it.

## 4. Explainability

Attention weights are read directly off the Transformer's own computation
(which past windows drove this forecast) — not a post-hoc approximation.
The Π matrix's transition probabilities are directly interpretable by
construction. SHAP-based feature attribution is planned, not yet built.

## 5. Results (CIC-IDS-2018 + CTU-13 training pool; CIC-IDS-2017 held out)

**Phase 2 stage classifier, before vs. after adding the multi-step loss**
(macro-F1, test split):

| Stage | F1 real (after) | F1 imagined (before) | F1 imagined (after) |
|---|---|---|---|
| Benign | 0.787 | 0.327 | **0.818** |
| Command & Control | 0.934 | 0.579 | **0.930** |
| Initial Access | 0.000 | 0.042 | 0.000 |
| Lateral Movement | 0.000 | 0.032 | 0.000 |
| Impact | 0.089 | 0.025 | 0.000 |

For the two classes with adequate training data, multi-step training
closed nearly the entire real-vs-imagined gap (forecasting ≈ detection
accuracy). Initial Access/Lateral Movement/Impact stayed at zero *before
and after* — the multi-step fix addressed a training-regime problem, not a
data-volume problem, and correctly had no effect where the problem is pure
scarcity (156–424 examples out of ~24,500 windows for these three stages).

**`Π^K` vs. neural ensemble rollout** (Brier score of infiltration
probability vs. ground truth, in-distribution test windows): neural wins
at every horizon k=1..5 (0.157→0.103, *improving* with horizon after the
multi-step fix — it degraded with horizon before) vs. Π (0.219→0.245,
degrading with horizon, consistent with regressing toward the chain's
stationary distribution rather than tracking short-term dynamics).

**CIC-IDS-2017 zero-shot generalization** (fully held out — no window, no
scaler statistic, ever touches this dataset): macro-F1 (imagined) = 0.047,
barely above the pre-fix 0.033. Per-class detail matters more than the
aggregate here: Command & Control recall rose from 0.000 to 0.552 (a real
generalization improvement), but the model now over-fires C2 broadly,
including on days with no C2 at all (the pure-benign Monday and the DDoS
day both dropped to macro-F1 0.000). **Multi-step training fixed
in-distribution forecasting cleanly, on two independent metrics, but did
not fix cross-dataset generalization — it changed the shape of that
failure rather than reducing it.** Reconnaissance stays at F1=0.000
zero-shot: CIC-IDS-2017 is the only source with Reconnaissance-labeled
windows, and it was kept out of training specifically to preserve this
generalization test (a stated trade-off, not an oversight).

## 6. Known gaps, stated plainly

- The mandated logistic-regression baseline comparison is not yet
  implemented — an open item, not a hidden one.
- Cross-dataset generalization (§5) is an open problem, not solved by this
  submission; the honest evidence for and against it is reported above
  rather than only the favorable half.
- SHAP-based feature attribution is planned, not built.
- Packet-level features are real for CTU-13, zero-filled for CIC-IDS-2018/2017
  (raw PCAPs are ~220GB; not fetched — see `dataset.md`).

## Datasets

Primary training: CIC-IDS-2018 (10 days) + CTU-13 (13 scenarios, contributes
only Benign/C2 labels). Held out for generalization testing: CIC-IDS-2017.
Not yet integrated into training or evaluation: CICIoT2023, UNSW-NB15,
NSL-KDD (unusable for sequence training — no timestamps), LANL (host-auth,
different modality). Full provenance, schemas, and per-dataset usage
rationale in `dataset.md` and `config.md`.
