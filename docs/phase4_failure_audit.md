# Phase 4 Stage A: Read-Only Failure Audit

**Date:** 2026-08-26
**Status:** Complete; no training performed

## Scope

The audit loaded all 15 completed checkpoints and recorded case-level outputs
for the 126-case development suite across five seeds (630 records). It did not
create a sealed suite, change a checkpoint, or train a new model. It compared
the matched generative operation, discriminative operation, multitask
operation, trained binding head, and a newly refit closed-form binding probe on
the frozen multitask response-boundary states.

## Track results across all seeds

| Track | Generative operation | Operation-only | Multitask operation | Multitask binding | Frozen binding probe |
| --- | ---: | ---: | ---: | ---: | ---: |
| Familiar forms/new operands | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Held-out templates | 55.0% | 49.4% | 56.7% | 71.7% | 72.2% |
| Lexical shift | 1.1% | 25.6% | 40.0% | 100.0% | 100.0% |
| Syntax/order reversal | 54.4% | 65.6% | 61.1% | 75.6% | 74.4% |
| Discourse distractor | 33.3% | 28.9% | 26.7% | 93.3% | 93.3% |
| Minimal contrast | 33.3% | 33.3% | 40.0% | **0.0%** | **0.0%** |

## Findings

The discourse gate violation is a seed-stability failure, not a uniform
regression. Seeds 17, 29, 53, and 67 all tie the baseline at 33.3% multitask
operation accuracy. Seed 41 alone falls from 33.3% to 0%, producing the -6.67
pp mean regression. Within seeds, discourse policies are deterministic across
all six operand pairs. For example, seeds 17/29/53 map every requested
SUBTRACT case to ADD and every COMPARE case to SUBTRACT.

The minimal-contrast binding failure is stronger and fully systematic. Every
one of the 90 cases is canonically A-first, and every multitask checkpoint
predicts B-first for every case. Refitting a closed-form linear binding probe
on frozen training states also predicts B-first for all 90. The necessary
invariance is therefore absent from the response-boundary representation; the
failure is not just a poorly optimized output head.

Across all tracks, the largest operation confusion remains SUBTRACT→ADD (114
cases) followed by SUBTRACT→COMPARE (62). This reinforces the Phase-2 finding
that surface scaffolds select competing operation policies rather than a
single intent boundary.

## Decision

Do not tune the completed development suite or weaken the gate. Register a
fresh-development 2 × 2 multitask screen that varies binding-loss weight and
fixed-budget scaffold counterbalancing. Reduced binding weight tests whether
the discourse collapse is harmful auxiliary-gradient variance; scaffold
counterbalancing tests whether the systematic binding hole can be repaired in
the representation. Preserve 50/50 mention order within every operation and
operand cell. A passing arm still requires a new sealed confirmation.

Artifacts:

- `runs/phase4_bottleneck_screen_v1/failure_audit_records.json`
- `runs/phase4_bottleneck_screen_v1/failure_audit.json`
