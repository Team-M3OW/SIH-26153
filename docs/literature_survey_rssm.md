# Literature Survey: Does Real RSSM Help Our World Model?

Scope: not "is RSSM well-established" (it is) — specifically, does adopting
RSSM's posterior/prior split and KL balancing solve a problem we actually
have, given what's already verified this session (our multi-step loss
works; a fancier encoder made us less stable; domain-specific heads, not
more shared-representation machinery, fixed our domain-generalization
problem). Claims below are verified from primary sources where marked;
one claim (§4) is reasoned, not directly quote-verified, and is flagged.

---

## 1. KL balancing solves a DIFFERENT problem than the one we already fixed

Fetched DreamerV3 directly (arXiv:2301.04104). The actual loss:

```
L_dyn = max(1, KL[sg(q(z_t|h_t,x_t)) || p(z_t|h_t)])
L_rep = max(1, KL[q(z_t|h_t,x_t) || sg(p(z_t|h_t))])
```

The paper states this exists to "avoid a degenerate solution where the
dynamics are trivial to predict but fail to contain enough information
about the inputs" — **representation collapse**, where the posterior and
prior trivially agree by encoding nothing useful. This failure mode exists
*only because* a separate posterior and prior exist to potentially collapse
into each other. **It cannot occur in our architecture** — we have one
deterministic encoder, no separate posterior network, nothing to collapse.
Adding the posterior/prior split would import a new failure mode and then
require KL balancing to fix it — solving a problem created by the same
change, not a stability gain on top of what we have.

Separately: PlaNet's explicit "latent overshooting" (multi-step KL penalty
between predicted and filtered-posterior latents) does **not appear in
DreamerV3's documented loss** — the full KL-loss section found above is
one-step, applied per timestep over a teacher-forced sequence, not an
explicit multi-step overshooting term. The Dreamer lineage gets multi-step
consistency a different way: unrolling the RSSM over full real sequences
during world-model training, plus (separately, policy-specific) BPTT
through multi-step *imagined* rollouts during actor-critic training. The
second mechanism doesn't apply to us (no policy). The first is mechanically
close to what our own multi-step open-loop loss already does, just scored
in decoded-observation space via Gaussian NLL instead of in latent space
via KL divergence against a posterior we don't have. **We are not missing
latent overshooting — we already have a working, validated substitute for
the one part of it that applies to a no-policy system.**

## 2. No verified example found of RSSM used with zero action/policy component

Searched specifically for RSSM/Dreamer components reused in pure
forecasting or anomaly detection with no action variable and no
actor-critic loop. Nothing found that names a specific, checkable paper
doing this — only generic, uncited claims ("RSSMs are applied in
forecasting") with no paper attached. **This is a negative result: the
objection that RSSM's posterior/prior split exists to serve imagined-policy
learning is not undercut by any counter-example found.** If real
precedent exists, it wasn't surfaced by this search; treat the objection as
standing, not settled.

## 3. DreamerV3's robustness claim is conditioned on data volumes far beyond ours

Verified exact per-domain budgets from the paper: Atari 200M frames,
ProcGen 50M, DMLab 100M, Minecraft 100M, Proprio Control 500K steps,
Visual Control 1M steps, and even the explicitly "sample-efficient" setting
(Atari100k) uses **400,000 environment frames** — the smallest budget in
the entire paper. Our total dataset is ~13,000 windows, train+val+test
combined. Atari100k alone is ~30x that; most of the paper's benchmarks are
4,000-15,000x our data volume. **The paper contains no low-data, small-model,
or small-task-count analysis at all** — its "one robust config" claim has
never been tested anywhere near our regime, and there is no basis in this
paper for assuming it transfers down by multiple orders of magnitude.

## 4. Does the posterior make domain-specific shortcut learning worse? — reasoned, not directly source-verified

A specific paper found in search (arXiv:2604.25416, "Biased Dreams: Limitations
to Epistemic Uncertainty Quantification in Latent Space Models") looked
directly on-topic but exceeded fetchable size and could **not** be read —
flagging this honestly rather than citing it unverified. Absent that
source, the mechanistic argument stands on first principles only: a
posterior network is optimized in part to reconstruct/predict well *given
the real observation it's conditioned on* — it has a strictly easier
optimization path to lean on observation-specific (and therefore
domain-specific) detail than a design where nothing downstream of the
transition ever sees the real future during either training or inference.
Combined with the already-verified finding (companion survey,
`literature_survey_domain_generalization.md`) that even a sophisticated
shared/adversarial/MoE architecture (CEPHALON) only reaches cross-domain
transfer via few-shot adaptation, not zero-shot invariance: adding a
posterior gives the model one more available shortcut in exactly the
setting where shortcuts have already been shown to be the dominant risk.
This is a plausible, mechanistically-grounded concern, not a proven one.

## 5. Yes — RSSM's deterministic recurrence is the correct substrate for the Koopman constraint, if that direction is retried

This doesn't need new search: Koopman Dreamer (arXiv:2607.19719, already
verified earlier this session) augments DreamerV3 by replacing **only** the
deterministic GRU recurrence with the spectrally-constrained Koopman
operator, keeping the stochastic/posterior machinery untouched. Our own
attempt added the Koopman operator *additively* on top of a Transformer's
output — a different, less faithful setup than what the verified ablation
actually tested. **If the Koopman direction is retried, doing so on a
standalone GRU-style deterministic recurrence (not a Transformer, not
additively) would be a more faithful reproduction of the one ablation
we're relying on as evidence** — but this only requires borrowing RSSM's
deterministic half, not its posterior/prior/KL-balancing apparatus.

---

## Recommendation

**(c), with a scoped exception matching part of (b).** Full RSSM adoption
is not justified: its defining addition (the posterior/prior split) fixes
a failure mode (representation collapse) that cannot occur in our
architecture, has no verified precedent of being used without a policy
loop we don't have, and is validated at data scales 30-15,000x larger than
ours with zero low-data analysis to lean on. It also plausibly (though not
directly verified here) hands the model an easier shortcut-learning path
in exactly the place (domain-specific overfitting) we've already proven is
the dominant risk.

The one piece worth taking is narrower than "adopt RSSM": **if the Koopman
spectral-constraint direction is retried, implement it on a standalone
deterministic recurrence (a GRU, matching Koopman Dreamer's actual design)
instead of additively on top of the Transformer.** That is borrowing one
component (a deterministic recurrent path) for a specific, already-evidenced
reason (faithfully reproducing the one verified ablation), not adopting
RSSM's architecture wholesale.
