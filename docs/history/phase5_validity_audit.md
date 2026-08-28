# Phase 5 validity audit — 2026-08-27

## Decision

Pause further capacity training. The pretrained-agent pipeline has confirmed
training and evaluation defects, so its previous readiness failures cannot
establish an architectural limit. Retain all old artifacts as diagnostics.
Repair and validate the benchmark, then evaluate the existing frozen weights
before considering retraining. The compact-parser results from Phases 0–4 use
a different pipeline; this audit does not invalidate their sealed results.

## Measured defects

The byte-identical 135M/360M corpus has 184,000 trajectories. The saved audit is
`runs/smollm2_360m_capacity/corpus_audit.json`.

| Finding | Consequence |
| --- | --- |
| All 16,000 missing-evidence and 8,000 evidence-disabled demonstrations stop at SEARCH, without an observation or final answer. | The models were never taught completion of these retrieval-abstention trajectories. Their 0% score does not isolate architectural inability. |
| All 10 sampled counterfactual tasks reference absent documents. | The evaluator asks for grounded answers that cannot be obtained from its world. |
| Multi-hop has zero training or evaluation cases. | An overall score does not cover the declared multi-hop suite. |
| All 10 evidence-disabled cases retain a nonempty document collection in the legacy evaluator. | A flag in task metadata does not actually withhold evidence. |
| All 10 language cases repeat just two training questions; 7/10 invariant and 5/10 direct-computation questions also occur in training. | These tracks measure reproduction plus some transfer, not clean unseen-form generalization. Question overlap alone is not leakage for retrieval tasks with different worlds. |
| The legacy grounded score accepts any one required citation and does not require it to have been observed. | Retain the historical score for replay; additionally require all proof lines to be cited and read. |
| 135M's configured seed was not enforced; 360M's was. | The exposure-matched single runs are not a seed-controlled causal capacity experiment. |

There is no train/evaluation **world-ID** overlap. That does not repair the
question overlap or missing evidence above. The 100-case sample covers only
2–10 worlds per suite; task counts must not be represented as independent
training repetitions or a reliable extrapolation to general-purpose use.

## Root causes and repairs

- `synth/generate.py` serialized each evaluation world **before** task creation
  added counterfactual documents. Serialization now occurs afterward.
- The insufficient-evidence trajectory branch appended only PLAN and SEARCH.
  It now records the live search observation and a terminal abstention.
- Multi-hop selected settlements by a nonexistent `region_id` entity property.
  It now uses canonical `inside` facts, includes both membership proofs, and
  skips tied populations rather than inventing a higher one.
- `synth/evidence.py` provides a non-mutating evidence-disabled/closed-book
  world view. Synthesis and the corrected frozen-evaluation wrapper now use it;
  the legacy runtime remains available explicitly for historical replay.
- SFT loading now refuses unfinished demonstrations before tokenization. New
  canary tests also exposed oracle-label copying into retrieval plans and
  direct-computation observations. Those plans now describe the procedure;
  direct-computation observations come from the actual restricted evaluator,
  and absent expressions raise an error instead of falling back to the label.

The latest repair pass passed 208 unit tests, including tests that reproduced
the actual missing-document, unfinished-trajectory, and empty-suite defects.
Historical corpora and checkpoints were not regenerated or overwritten.

The fresh repair corpus has zero missing required proof lines and includes all
eleven declared suites. Its audit's zero training-question overlap is relative
to its **empty evaluation-only training file**, not the pretrained runs' old
training corpus. Familiar language/math forms remain an explicit limitation;
this probe is not a test of broad unseen-form generalization.

Remaining known limits before a new training experiment include teacher search
hit injection, fabricated recovery observations, fallback operand constants,
and silent trajectory truncation. A green code suite does not certify these
scientific controls. Do not call the full training pipeline validated until
tool-observation replay and tokenized-target coverage checks pass.

## Repaired probe and value-dependence check

The frozen 135M checkpoint finished the repaired 110-case probe with 53/110
(48.18%) strict grounded success. Counterfactual success is 9/10, showing that
the previous 0/10 with missing evidence was not a reliable measure of model
capability. This is a different task sample and not a causal before/after
estimate. Multi-hop and both retrieval-abstention tracks still score zero.

Counterfactual target values were themselves fixed in the old training corpus.
To test actual evidence dependence, `runs/phase5_grounding_value_swap_v1` holds
twenty preregistered variants of ten counterfactual cases: each pair has the
same question and document identifiers, but different newly generated answers
in the referenced document. Both variants must be strictly grounded and
correct for a pair to pass. No model output selects the new values. This is a
frozen-weight development diagnostic, not a sealed architecture confirmation.

Both model configurations for this probe are marked `evaluation_only`; the
trainer refuses those configurations before loading weights or creating run
artifacts. Historical and repaired-corpus model weights remain untouched.

**Completed result:** 135M passed 8/10 entire pairs and 360M passed 9/10;
individual-case strict success was 17/20 and 18/20. Neither reused an old
counterfactual or real-world answer. Both abstained on the same pair, and 135M
made one novel-name copying error. This supports evidence-conditioned behavior
on the tested cases, not general-purpose competence. See the
[current feasibility opinion](phase5_feasibility_verdict.md). Final code
verification: 217 unit/integration tests passed.

## Next discriminating measurement

1. Finish only the useful legacy replay/calibration; label it diagnostic.
2. Generate a fresh evaluation-only corpus with full required evidence and
   explicit suite coverage (`configs/phase5_validity_replay.yaml`). This is a
   repair/development probe, not a newly sealed architectural confirmation.
3. Apply evidence controls in the runtime; validate terminal trajectories,
   proof completeness, and tool-observation consistency before training.
4. Evaluate the same frozen 135M/360M checkpoints with both legacy and strict
   grounding summaries on identical repaired tasks. Report per-suite breadth
   and task/world-conditional uncertainty, not seed confidence.
5. If failures persist, test a small corrected-trajectory control before
   increasing model size. Full architectural ranking still requires a clean
   scratch baseline and genuinely shared held-out semantic/agent tracks.
