# Literature Survey: Alternative World-Model Lineages, Evaluated for Stability

Scope: not "what's most accurate on a big benchmark" — what's most likely to be
**stable** on a small (~13,000 window), heterogeneous (3 network environments),
tabular (43-dim flat feature vector) dataset, given the specific instability
we just measured (a GraFT-style R-GCN encoder gave wildly different,
sometimes worse-than-random results between a 30-epoch and an 80-epoch run of
the *identical* setup). All papers below were fetched and read in full
(not trusted from search snippets), current as of Sept 2026.

---

## 1. JEPA-style world models — NOT a free stability win

The framing "JEPA removes reconstruction, so it should be more stable" is
contradicted by JEPA's own current literature. Multiple 2026 papers exist
**specifically to fix JEPA's own documented instability**:

- **LeWorldModel** (arXiv:2603.19312, 286 votes) opens by stating plainly that
  "existing [JEPA] methods remain fragile, relying on complex multi-term
  losses."
- **Sub-JEPA** (arXiv:2605.09241) exists because "JEPA training is subject to
  a bias-variance trade-off" requiring a new "Subspace Gaussian
  Regularization" just to reach stable training.
- **JEPA-x** (arXiv:2608.24044) names the mechanism directly: "the encoder
  and predictor are optimized jointly and can co-adapt to lazy shortcuts" —
  i.e. **representation collapse**, JEPA's own signature failure mode.
- **Phys-JEPA** (arXiv:2606.16076, read in full) is the most directly
  relevant: a JEPA world model for multivariate time-series forecasting.
  Its own "Extended Robustness Checks" section reports: *"full Phys-JEPA was
  less robust in the shortened training setting, suggesting sensitivity to
  optimization and loss weighting"* — and states outright that "JEPA serves
  as a reasonable temporal representation backbone but **does not
  universally outperform a supervised baseline**." Getting it to work at all
  required adding a physics-informed decomposition, a stop-gradient target
  encoder, an EMA teacher, and a dedicated anti-collapse regularizer
  (SIGReg) — strictly more moving parts than our current model, not fewer.

**Verdict:** removing the decode/reconstruct loss doesn't remove instability,
it *trades* our failure mode (small-encoder training variance) for JEPA's
own well-documented one (representation collapse), which itself needed new
regularizers invented in 2026 to control. High implementation risk (target
encoder + EMA teacher + collapse regularizer), no evidence it would be more
stable for us specifically. **Not recommended.**

## 2. Mamba/SSM as the transition backbone — real, but not the actual lever

Real work exists (**GEM**, arXiv:2605.07326, a Mamba-based LiDAR world
model for autonomous driving), and Mamba-3 (arXiv:2603.15569, CMU/Princeton/
Together AI/Cartesia) confirms the lineage is active and credible. But
nothing found shows Mamba/SSM backbones are specifically *more stable* than
attention for small-data multi-step rollout — the stability evidence in
this space points somewhere more specific and more useful (§3).

## 3. The actual mechanistic answer: constrain the *transition*, not the encoder

**Koopman Dreamer** (read in full) is the single most relevant paper found,
and it reframes what we should actually fix. Its stated motivation
describes our exact symptom: *"[generic neural latent transitions] do not
typically offer direct mechanisms to control properties such as modal
persistence, contraction, or oscillation... making the long-horizon
behavior of the model difficult to regulate and diagnose"* — i.e. an
unconstrained transition network's error growth over a multi-step rollout
is fundamentally unpredictable, exactly the "30 vs 80 epochs flips the
result" signature we measured.

Its fix: constrain the deterministic part of the transition to a
**block-diagonal rotation-scaling structure with a bounded spectral
radius** (a Koopman-operator parameterization: `ϕ_{t+1} = A_K·ϕ_t + ...`
where `A_K`'s eigenvalues are explicitly clamped into a controlled range).
This gives a **provable multi-step error bound**
(`‖error‖ ≤ κ*^H‖error_0‖ + Σ κ*^{H-1-i}δ_i`, `κ* = spectral radius +
Lipschitz term`) instead of hoping training happens to find stable dynamics.

Empirically: 89.4% reduction in long-horizon latent MSE vs. vanilla
DreamerV3 on DeepMind Control Suite, beat DreamerV3 on 8/9 tasks, and an
ablation removing *only* the spectral constraint (keeping everything else)
dropped mean return by 20.4% with visibly noisier rollouts — direct
evidence the spectral constraint itself, not the rest of the architecture,
is what buys the stability.

**Why this is the right lever for us specifically:** our instability showed
up in multi-step rollout behavior (Brier/AUC swinging wildly with training
length), and neither the flat MLP nor the graph encoder ever constrained
*how the transition's own errors grow over K steps* — that part of our
architecture (the causal-Transformer Gaussian head) has always been
unconstrained. We changed the encoder twice and never touched the actual
mechanism the instability signature points at.

**Implementation for us:** small, surgical, and it doesn't require
abandoning anything — replace (or add alongside) our current
`trans_head` output with a spectrally-bounded linear/rotation component for
the deterministic part of the transition, keep the stochastic Gaussian
sampling as-is. Pure PyTorch, no new dependency, no encoder change needed.
Real implementation effort (needs the rotation-block parameterization and
the multi-step training objectives that keep training/imagination
consistent), but it's targeted at the actual diagnosed failure, not a
guess.

## 4. Contrastive/non-reconstructive objectives (CPC, SimCLR-family) — real risk factor for small data

CPC (van den Oord et al. 2018, 14,322 citations — verified foundational) is
real and does show representations transfer well data-efficiently in some
domains. But the field's own literature flags a direct risk for us:
**SimCLR's own paper states contrastive learning "benefits from larger
batch sizes and more training steps"** — the negative-sampling/contrastive
mechanism needs scale to work well. We have ~13,000 windows total, several
attack classes at 150-450 examples. Supervised Contrastive Learning does
report being "more stable to hyperparameter settings" in its own domain
(image classification) — some genuine evidence for robustness — but that
evidence doesn't transfer automatically to tabular/small-N regimes, and the
batch-size dependency is a concrete, documented obstacle specifically for
our data scale. **Not recommended as a near-term change** — real rebuild
cost, with a well-documented failure mode (needs scale) that we don't have.

## 5. Tabular-ML-specific recipes — the field's own current answer is "keep it simple"

Directly relevant, current (2025-2026) findings: **TabM** is described in
its own literature as "an efficient ensemble of MLP models" for tabular
data; **RealMLP** is "an MLP enhanced with robust scaling, numerical
embeddings" — i.e., the tabular-ML field's own state of the art for
small/medium tabular data is *simple architectures with proven small
enhancements* (robust scaling — which we already do — and lightweight
ensembling), not complex graph/attention rebuilds. Separately, "Tabular
Data: Is Deep Learning all you need?" and related 2024-2026 work
consistently finds well-regularized MLPs remain highly competitive with
much fancier tabular architectures. This is direct evidence *for* keeping
our flat MLP encoder rather than continuing to chase encoder complexity.

**A concrete, low-risk idea this suggests that we haven't tried:** a small
ensemble of 3-5 independently-initialized flat-MLP-encoder world models,
averaging their rollout predictions — cheap, no new architecture, and
ensembling is specifically the technique the tabular-ML field currently
relies on for robustness at our scale, rather than a single larger/fancier
model.

## 6. Is our GraFT/R-GCN instability consistent with documented GNN failure modes?

Yes, concretely. Verified: **"gradient oversmoothing" specifically prevents
optimization during training** (2025 finding, distinct from the older,
better-known representation-oversmoothing), and GNN performance is
documented as **sensitive to initialization** near the
oversmoothing/non-oversmoothing transition — consistent with our own
30-vs-80-epoch swing. The standard mitigations (residual connections,
normalization layers) are real but **the evidence is explicitly mixed** —
"their effects to overcome oversmoothing are diminished by increasing depth
or growing dataset size" — so adding them (our current `GraphEncoder` has
neither) is worth trying but isn't guaranteed to fully resolve the
instability on its own.

---

## Recommendation

**Don't chase a bigger encoder rebuild (JEPA, contrastive, another GNN
pass). Two concrete, low-risk changes, in order:**

1. **Add a Koopman-style spectrally-constrained transition function** (§3)
   — this is the one change directly targeted at the actual mechanism our
   instability points to (unconstrained multi-step error growth), backed by
   a real theoretical bound and a real ablation proving the spectral
   constraint specifically (not the rest of the architecture) is what buys
   stability. Keep the flat MLP encoder — it's already proven, and current
   tabular-ML literature (§5) independently supports staying simple there.
2. If more robustness margin is wanted cheaply, **ensemble 3-5 flat-MLP
   world models** (§5) rather than building one more complex one — this is
   the tabular-ML field's own current answer to small-data robustness, at
   near-zero new implementation risk.

The graph-encoder attempt wasn't wrong to try, but the evidence now points
at the transition function, not the encoder, as where our actual
instability lives — every encoder we've used (flat MLP, R-GCN) sits
upstream of an unconstrained transition, and Koopman Dreamer's own ablation
is direct evidence that's the component that determines multi-step rollout
stability.
