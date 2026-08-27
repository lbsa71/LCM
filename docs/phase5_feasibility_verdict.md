# LCM feasibility checkpoint — 2026-08-27

## Current opinion

**Grounded procedural behavior is feasible at the tested pretrained scale.
General-purpose readiness is not demonstrated. Keep the 135M pretrained route
as the practical baseline and improve supervision/control before adding
parameters.** This is a research allocation decision, not a proven ranking of
all architecture families or a prediction that a general-purpose agent is
impossible.

The original negative evidence overstated architectural limitations because
the pretrained benchmark had missing counterfactual documents, no multi-hop
cases, and 24,000 unfinished abstention demonstrations. Those findings changed
the next experiment: repair validity and inspect frozen weights first.

## What the completed measurements establish

| Measurement | SmolLM2-135M | SmolLM2-360M |
| --- | ---: | ---: |
| Historical-data replay, 100 tasks | 45% strict grounded | 41% strict grounded |
| Repaired development probe, 110 tasks | 48.18% | 50.00% |
| New-value probe, 20 cases | 17/20 correct and grounded | 18/20 |
| Both variants correct, ten identical-question pairs | 8/10 pairs | 9/10 pairs |
| Reuse of old counterfactual or real-world answers in value swaps | 0/20 | 0/20 |
| Reported training-loop time | 26.30 minutes | 8.50 hours |

The new-value probe changes document answers while retaining each pair's
question and document identifiers. Getting both answers right requires
responding to the changed evidence rather than returning a fixed memorized
answer. Both models fail the same pair by abstaining before retrieving; 135M
also makes one novel-name copying error. This is positive evidence for the
central procedural-grounding idea, on a deliberately small diagnostic.

The repaired capacity difference is only +1.82 percentage points, with a
world-cluster interval of [-2.20, +5.00] conditional on these fixed runs and
sampled worlds. It does not establish a reliable overall gain. The 360M run
cost 19.39 times as much reported training time; its improvement on recovery
and one extra value-swap pair does not yet justify further size increases.
The cause of that training slowdown is not identified.

Both models still score zero on multi-hop and retrieval-abstention in the
repaired probe. Language/direct-math successes include familiar forms, and
the old training seed is unverified. These are not five-seed, architecture-wide
learning curves. The repaired 48–50% aggregate is not general-purpose readiness.

## Architecture interpretation

| Approach | What is supported | What remains unproven |
| --- | --- | --- |
| Pretrained model + tools | Most useful current baseline; small models can follow newly supplied document values. | Reliable abstention, composition, broader language/task transfer, and convergence rate. |
| Compact semantic parser → typed representation → controller | The tested 96-wide branch has repeated robustness failures, including sealed confirmation. | Whether a larger/pretrained semantic parser with a constrained controller succeeds; the small-parser result cannot reject that family. |
| From-scratch ReAct | Historical artifacts remain useful diagnostically. | A corpus-validated comparison; the old 300M score cannot rank this family against the others. |

Consequently, there is **no defensible timeline to a general-purpose agent**
from these results. The measured capacity axis has no reliable overall payoff,
and the earlier training-token slope used a defective benchmark. Continuing
unchanged scaling is not justified. Equally, these failures do not establish
that any architecture can never work.

## Next experiment and outstanding work

Use a small, matched, five-seed complete-trajectory/control experiment on 135M,
with present/absent/withheld evidence counterbalanced, real tool-observation
replay, and terminal-target coverage checked before training. The design
direction is in [the next-control note](phase5_next_control.md); it is not yet
registered or trained. A repaired learning curve, clean scratch comparator,
and broader shared architecture benchmark remain outstanding. The full
research goal is therefore **not complete**.

## Evidence and verification

- `runs/phase5_capacity_comparison_legacy.json`
- `runs/phase5_capacity_comparison_repaired.json`
- `runs/phase5_grounding_value_swap_v1/registration.json`
- Each value-swap model directory's `paired_probe_analysis.json`, frozen
  manifest, full episode ledger, and evaluation timings.
- [Validity audit and repairs](phase5_validity_audit.md).
- [Capacity experiment and provenance caveats](phase5_pretrained_capacity.md).

This round completed 460 scored frozen-model episodes plus an explicitly
unscored partial old-data milestone. All checkpoints were retained. No new
training was launched. Verification: **217 unit/integration tests passed**, with
one existing Starlette deprecation warning; `git diff --check` is clean.
