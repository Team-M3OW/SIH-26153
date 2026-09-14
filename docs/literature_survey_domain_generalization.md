# Literature Survey: Domain-Conditioned Architectures for Small, Heterogeneous, Tabular Multi-Domain World Models

Scope: not generic domain generalization — specifically, architectures that
share a common backbone (our encoder+transition) while acknowledging domains
differ elsewhere, evaluated against our exact failure mode: pooling 3 real
network-traffic datasets causes `Π`/classifier to work on whichever domain
dominates and be worse-than-random on the others; domain-adversarial training
gave a real but partial fix (helped the worst domain, hurt the previously-best
one). All claims below were verified by fetching primary sources (including
reading a full 8-page PDF, not trusting search snippets).

---

## 1. FiLM / conditional modulation — mature mechanism, low implementation risk, no domain-id-specific precedent found

FiLM (Perez et al. 2017, feature-wise affine modulation conditioned on an
external signal) is a decade-mature, trivially-implementable mechanism (two
small linear layers producing a per-feature scale and shift). Verified
current use: **"Feature-aware Modulation for Learning from Temporal Tabular
Data"** (arXiv:2512.03678, NeurIPS 2025, read in full) — conditions feature
representations on *temporal* context to handle within-source concept drift,
**not** cross-domain generalization. No paper was found conditioning
FiLM-style modulation on a *domain-id* specifically for tabular/DG (the
mechanism trivially generalizes -- swap the conditioning input from a time
embedding to a domain embedding -- but this specific application isn't a
citable, validated result, just a plausible adaptation).

**Verdict:** legitimate, cheap thing to try (a domain-conditioned FiLM layer
between the encoder and transition), but going in with no direct evidence it
solves *our* specific one-domain-vs-another trade-off, only that the general
mechanism is sound and low-risk to implement.

## 2. Domain-specific heads with a shared trunk — real NIDS-specific precedent, but as ensembling, not per-domain output heads on one network

Found real, if only snippet/abstract-level verified, applied work: a
**Heterogeneous Deep Stacked Ensemble (HDSE-IDS)** where "each base model is
trained on a distinct dataset and **frozen** to preserve domain-specific
decision boundaries, while a meta-learner aggregates their predictions,"
reporting >95% cross-domain accuracy. This is architecturally different from
"one network, domain-specific final layer" -- it's full per-domain models,
frozen, combined by a learned aggregator. A second paper (domain-adaptive
multi-modal IDS across heterogeneous networks, arXiv:2508.03517) targets the
same applied niche. Neither was read in full (abstract/snippet level only --
flagged honestly).

## 3. Domain-conditioned mixture-of-experts — real example exists, but the gate is input-conditioned, not domain-conditioned

**CEPHALON** (arXiv:2606.29339, read in full via the PDF) is a real,
verified multi-domain architecture: shared trunk (sparsely-gated channel
mixture-of-experts + an adaptive-topology graph module) + a domain-adversarial
branch. Important correction to the initial hypothesis: CEPHALON's MoE gate
is `g = softmax(W_g . pool(x))` -- conditioned on the **input window**, not
on the domain label directly -- and its "tri-cardiac heads" are **redundant
copies for fault tolerance**, not domain-specific heads. So CEPHALON is
*not* actually an example of combining domain-specific heads with
domain-adversarial training (the thing point 5 asked about); it's
domain-adversarial invariance plus input-conditioned MoE plus redundancy,
evaluated with per-domain reporting, not per-domain output layers.

**The most important verified finding from this paper, applicable directly
to us:** despite this fairly sophisticated multi-domain design, the authors
report plainly that **"zero-shot cross-domain transfer fails, and the value
of the shared representation is realized only through few-shot
adaptation."** Even a shared trunk + domain-adversarial branch + MoE gating
could not achieve zero-shot generalization across their three physically
distinct domains (seismic / DAS / industrial vibration) -- it needed a small
amount of labeled target-domain data to actually transfer. This is a direct,
verified precedent for what we measured: domain-adversarial invariance alone
is a real but partial fix, not a path to zero-shot cross-domain success, in
a paper built specifically to test exactly this question.

## 4. Meta-learning / episodic DG (MLDG-style) — verified, not usable at our domain count

Confirmed limitation, directly on point: with few source domains, "each
source domain takes turn to be meta-test domain, which means the model can
*peek at* the domain it's told to generalize to during training after some
iterations" -- with only 3 domains total (our exact situation), this leakage
effect is severe, since the same 3 domains cycle repeatedly through
meta-train/meta-test roles. MLDG is also MAML-derived and needs second-order
derivatives -- real added training cost for a technique whose own literature
says it's compromised at n=3 domains. **Not recommended** -- this is a
regime the technique's own documented limitation rules out, not just an
unproven fit.

## 5. Adversarial + domain-conditioning combined — the CEPHALON finding (§3) is the direct answer here

Re-reading point 5 in light of §3: the honest answer is that the *specific*
combination asked about (domain-specific heads AND domain-adversarial
training together) wasn't found as a validated technique anywhere in this
search. What *was* found, and is more useful: verified evidence that
adversarial invariance alone (even inside a more sophisticated shared
architecture than ours) caps out below zero-shot cross-domain success, and
that **few-shot target-domain adaptation, not more architectural
invariance-forcing, is what closed the gap in the one real system that
tested this directly.**

## 6. NIDS-specific cross-domain generalization literature — exists, and its actual working recipe differs from adversarial invariance

Real, current (2025-2026) applied niche confirmed via §2: the field's own
best-performing pattern for cross-dataset IDS generalization is **train
separate models per dataset, freeze them, combine via a meta-learner** --
not "train one adversarially-invariant shared representation." This is a
meaningfully different recipe than domain-adversarial training, and it's
specifically validated in our exact application domain (network intrusion
detection across heterogeneous datasets), which generic vision-domain
literature isn't.

---

## Recommendation

**Don't chase more shared-backbone invariance machinery (FiLM, domain-conditioned
MoE gating) as the next move — the verified evidence points somewhere more
specific: stop trying to make ONE set of final outputs (classifier, `Π`)
work for all three domains at all, and combine what we already have (the
domain-adversarial encoder) with domain-SPECIFIC final layers instead of
domain-INVARIANT ones.**

Concretely: keep the domain-adversarial-trained encoder+transition as the
shared dynamics backbone (it's already built, and CEPHALON's finding
suggests a *reasonably* transferable shared representation is achievable
even if not perfect) -- then, instead of one shared stage-classifier and one
shared `Π` fit across all pooled data, fit **domain-specific classifier heads
and domain-specific `Π` matrices** on top of that shared encoder, selected by
the (already-available) domain label at train and inference time. This is
directly the NIDS field's own verified pattern (§2, §6: per-domain final
decision layers on a common representation) rather than a generic
vision-domain-generalization import, is low implementation risk (we already
carry `domain_for()` / domain ids through the pipeline for the adversarial
training), and doesn't repeat the mistake of asking one small linear
boundary to serve three genuinely different traffic distributions at once --
which is the root cause of the original failure this whole investigation
started from.

If a genuinely new, unseen (4th) network needs to be handled at deployment
without a domain label, CEPHALON's own finding says the honest expectation
should be **few-shot adaptation** (a small amount of labeled data from the
new environment to select/fine-tune the right per-domain head), not
zero-shot success from adversarial invariance alone -- worth stating
plainly in the architecture doc rather than implying zero-shot
cross-environment deployment is solved.
