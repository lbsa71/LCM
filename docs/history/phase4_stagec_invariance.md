# Phase 4 Stage C: Paired Representation Consistency

**Date:** 2026-08-26
**Status:** Complete; sealed confirmation failed

## Question

Can direct same-intent consistency supervision create the lexical/syntax
invariance that Stage B's frozen encoders lack, while preserving the strong
minimal-contrast benefit of scaffold counterbalancing?

## Registered design

Use Stage B's strongest training condition: 25% scaffold replacement and
binding-loss weight 0.25. Re-evaluate its five checkpoints as the zero-weight
lead. Train two new paired five-seed cells:

| Same-intent consistency weight | Runs |
| ---: | ---: |
| 0.00 | Reuse five validated Stage-B checkpoints |
| 0.25 | Five new runs |
| 1.00 | Five new runs |

Within each operation/operand frame, pair every A-first form exactly once with
a B-first form. A pair always has the same operation and canonical operands.
Use eight pairs per update, hence 16 sequences per update, for the same 3,200
updates and sequence-token exposure as the reused lead. Both members receive
operation and binding supervision; mean-squared distance between their
log-softmax operation logits is the only new loss. Binding states are not
aligned.

## Fresh development and gate

Freeze six new operand pairs and 108 new utterances. Each of six tracks has 18
cases and every track/operation is exactly 3/3 A-first/B-first. Re-evaluate the
matched Stage-A generative baseline, the reused Stage-B lead, and both new
consistency arms.

Apply the unchanged gate against the matched generative baseline: at least
+10 pp worst-group operation accuracy, paired-bootstrap lower bound strictly
above zero, and no operation track regression over five points. Select at most
one arm by worst-group gain then macro gain. No sealed suite exists; create one
only after a gate pass.

Preparation validated all ten reused source checkpoints, the 144-pair manifest,
16 sequences per update, 3,200 updates, the 108-case fresh suite, and zero
train/development or Stage-B/Stage-C text overlap. The required unit suite
passed 177 tests before launch.

## Resume checkpoint — 2026-08-26

The five `consistency_025` seeds (17, 29, 41, 53, and 67) completed with
checkpoints, step metrics, and final metrics. The runner was stopped before any
`consistency_1` training completed. It had created
`consistency_1/seed_17/tokenizer`, but no metrics or checkpoint for that seed;
the resumable runner therefore treats it as incomplete and trains it normally.

Resume with the `screen-stagec` command below. Existing compatible completed
runs are validated and skipped, leaving the five `consistency_1` seeds. Apply
the registered gate only after all ten new runs and the aggregate analysis are
complete.

The remaining cell was resumed on 2026-08-27 after the full unit suite passed
177 tests and Stage-C preparation revalidated every source fingerprint and
registered control.

## Screen result — 2026-08-27

All ten runs completed. Relative to the matched generative baseline,
`consistency_025` gained 26.67 percentage points in worst-group operation
accuracy (paired-bootstrap 95% CI [+10.00, +41.11]) and 12.22 points in macro
operation accuracy (CI [+4.89, +22.45]), with no track mean regressing by more
than five points. It is the only arm that passes the registered gate.

The higher consistency weight gained 24.44 worst-group points (CI [+12.22,
+35.55]) but regressed syntax/order reversal by 6.67 points and fails. The
causal contrast remains unresolved: consistency-0.25 versus the matched
no-consistency lead changed worst-group accuracy by only +1.11 points (CI
[-7.78, +12.22]) and macro accuracy by +1.11 (CI [-7.11, +10.00]). Selection
therefore establishes a candidate configuration, not proof that the
consistency term caused its development-set advantage.

## Sealed confirmation registration

Evaluate frozen checkpoints only; do not retrain or tune. Compare the matched
generative baseline, the no-consistency lead, and selected
`consistency_025` checkpoints over the five paired seeds. The sealed suite has
six entirely new operand pairs and six fresh pressure tracks. Every
track/operation/operand frame contains both A-first and B-first wording: 36
cases per track and 216 cases total.

Apply the unchanged selection gate to the candidate versus the generative
baseline: at least +10 pp mean worst-group accuracy, a strictly positive paired
bootstrap lower bound, and no operation-track mean regression beyond five
points. Report candidate-versus-lead accuracy and paired-form operation
agreement/correctness as secondary mechanism diagnostics; these diagnostics
cannot override a failed primary gate. Open and evaluate this suite once.

## Sealed result — 2026-08-27

The candidate failed confirmation. Relative to the generative baseline,
worst-group operation accuracy changed by -2.78 pp (CI [-15.00, +10.00]) and
macro operation accuracy by -2.78 pp (CI [-11.11, +2.78]). Held-out templates
regressed by 32.78 pp (CI [-52.78, -13.34]), with an additional 9.45 pp mean
lexical regression. The primary gate is decisively not met.

The consistency candidate increased paired-form operation agreement over the
baseline by 9.56 pp (CI [+3.56, +14.22]), but improved the rate at which both
forms were correct by only 2.00 pp (CI [-4.44, +7.33]). Versus the
no-consistency lead, agreement changed by +12.44 pp with an interval crossing
zero, while paired correctness was effectively flat (+0.44 pp). This is
evidence for more consistent predictions without reliable semantic
correctness—consistent errors rather than useful invariance.

The compact 96-wide semantic-bottleneck configuration is rejected as a robust
general parser candidate. Its checkpoints and all 3,240 sealed case
predictions are retained for architecture comparison and failure analysis; the
sealed suite must not be reopened for tuning.

## Execution guard

```powershell
./.venv/Scripts/python.exe -m eval.semantic_bottleneck `
  --config configs/phase4_stagec_invariance.yaml --stage prepare-stagec

./.venv/Scripts/python.exe -m eval.semantic_bottleneck `
  --config configs/phase4_stagec_invariance.yaml --stage screen-stagec

./.venv/Scripts/python.exe -m eval.semantic_bottleneck `
  --config configs/phase4_stagec_invariance.yaml --stage prepare-confirm-stagec

./.venv/Scripts/python.exe -m eval.semantic_bottleneck `
  --config configs/phase4_stagec_invariance.yaml --stage confirm-stagec
```
