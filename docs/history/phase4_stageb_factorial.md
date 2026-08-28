# Phase 4 Stage B: Binding-Weight × Scaffold-Coverage Factorial

**Date:** 2026-08-26
**Status:** Complete; no arm passed the registered gate

## Question

Can the strong Stage-A multitask operation signal pass the unchanged robust
gate if we reduce harmful auxiliary-gradient pressure and explicitly break the
systematic scaffold-to-binding shortcut?

## Registered design

At K=8/R178 and the same five seeds, compare:

| Binding-loss weight | Ordinary 50/50 corpus | Scaffold-counterbalanced corpus |
| ---: | --- | --- |
| 1.0 | Reuse five validated Stage-A multitask checkpoints | Five new runs |
| 0.25 | Five new runs | Five new runs |

The matched Stage-A generative checkpoints are evaluation-only gate baselines.
All arms retain 288 examples, 3,200 updates, 320 warmup updates, the fixed byte
tokenizer, model, optimizer, train operands, and exact 50/50 A-first/B-first
balance within every operation/operand cell. An active scaffold factor replaces
one A-first and one B-first form per cell with training-only shared-scaffold
examples; it never appends examples.

## Fresh development suite

Stage A’s suite is audit-only. Stage B freezes six new operand pairs and new
utterances for familiar, held-out, lexical, syntax/order, discourse, and
minimal-contrast tracks. Every track and operation has three A-first and three
B-first cases. No development utterance may occur in either training corpus.

## Analysis and gate

Report paired binding-weight effects with and without scaffold coverage,
scaffold effects at both weights, and their interaction for operation macro,
operation worst group, binding macro, and joint macro. Compare every arm to
the re-evaluated matched generative baseline.

An arm advances only if it gains at least 10 pp worst-group operation
accuracy, the paired-bootstrap 95% lower bound is strictly positive, and no
operation track regresses more than five points. Select at most one passing arm
by largest worst-group gain, then macro operation gain. A passing screen permits
creation of a new sealed confirmation suite; no sealed text exists yet.

Preparation validated both 288-example corpora, exact 144/144 global binding
balance, 3/3 binding balance within every development track/operation, zero
exact train/development overlap, all ten reused checkpoint fingerprints, and
the 3,200-step/320-warmup schedule. The required unit suite passed 171 tests
before the screen was launched.

## Result

| Arm | Worst-group delta vs generative | Macro operation delta | Material trade-off |
| --- | ---: | ---: | --- |
| Weight 1.0, ordinary | -10.00 pp [-16.66, -3.33] | -9.33 pp [-15.33, -2.67] | Syntax -20.00 pp |
| Weight 1.0, scaffold | 0.00 pp [-11.11, +11.11] | +6.00 pp [+1.78, +9.78] | Minimal +33.34 pp; syntax -13.34 pp |
| Weight 0.25, ordinary | -10.00 pp [-16.66, -3.33] | -8.00 pp [-12.00, -5.33] | Syntax -18.89 pp |
| Weight 0.25, scaffold | +1.11 pp [-16.66, +15.55] | +12.67 pp [+7.33, +18.00] | Minimal +55.56 pp; syntax -10.00 pp |

No arm reached the required +10 pp worst-group gain with a positive lower
confidence bound and no track regression over five points. No sealed suite was
created.

Scaffold coverage has a real average effect: at binding weight 0.25 it adds
+20.67 pp macro operation accuracy (CI [+14.00, +27.33]) and +22.00 pp joint
macro accuracy (CI [+18.00, +25.11]). Its worst-group effect is +11.11 pp but
remains uncertain (CI [-1.11, +22.22]). Reducing the binding weight alone has
no reliable operation effect. The operation-level interaction is also
uncertain, although the joint interaction is +9.56 pp (CI [+3.78, +15.11]).

## Read-only failure and representation audit

The strongest arm reaches 88.89% minimal-contrast operation accuracy and
66.67% all-three-operation triplet consistency, versus 33.33% and 0% for the
matched generative baseline. Its remaining floor is seed-dependent lexical
shift and syntax reversal; `SUBTRACT` remains the weakest operation.

A closed-form linear probe refit on each frozen training representation does
not reveal a hidden readout solution. For weight-0.25 scaffold, robust operation
accuracy is 65.11% for the trained head and 63.78% for the probe. Lexical shift
is 35.56% versus 34.44%, syntax reversal 53.33% versus 51.11%, and the probe
repairs none of the head's errors. The next intervention must therefore shape
the representation during training rather than replace the final classifier.

## Decision

Retain scaffold counterbalancing and binding weight 0.25 as the strongest
development lead, but do not promote it. Register a paired same-intent
consistency objective at two weights, with opposite-order surface forms paired
inside a fixed sequence-token budget and a new development suite. This tests
representation invariance directly without returning to template-count,
repetition, or width scaling.

## Artifacts

- [Registration](../evidence/records/phase4_stageb_factorial_v1/registration.json)
- [Split manifests](../evidence/records/phase4_stageb_factorial_v1/split_manifests.json)
- [Raw screen results](../evidence/records/phase4_stageb_factorial_v1/screen_results.json)
- [Gate and factorial analysis](../evidence/records/phase4_stageb_factorial_v1/screen_analysis.json)
- [Case predictions](../evidence/records/phase4_stageb_factorial_v1/case_predictions.json)
- [Frozen-probe audit](../evidence/records/phase4_stageb_factorial_v1/frozen_probe_audit.json)

## Execution guard

```powershell
./.venv/Scripts/python.exe -m eval.semantic_bottleneck `
  --config configs/phase4_stageb_factorial.yaml --stage prepare-stageb

./.venv/Scripts/python.exe -m eval.semantic_bottleneck `
  --config configs/phase4_stageb_factorial.yaml --stage screen-stageb
```
