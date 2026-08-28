# Phase 4 Stage A: Discriminative Semantic Bottleneck Screen

**Date:** 2026-08-26
**Status:** Complete; no arm passed the registered gate

## Motivation

Phase 2.5 did not establish that typed semantic representations are harmful.
Its typed-frame arm changed two things at once: it required more semantic
content and required a small causal language model to autoregressively spell
the entire frame. This screen separates representation from serialization.

## Registered design

At K=8/R178 and the same five paired seeds, compare:

| Arm | Encoder objective | Outputs | New runs |
| --- | --- | --- | ---: |
| Matched generative baseline | Masked causal LM | `OP=...` byte sequence | 5 |
| Discriminative operation | End-to-end classification | Three-way operation logits | 5 |
| Discriminative operation + binding | End-to-end multitask classification | Three-way operation plus A-first/B-first logits | 5 |

The original K=8 corpus is not suitable for this test unchanged: it contains
252 A-first and 36 B-first examples, and every B-first example is SUBTRACT.
That makes binding both a majority-class shortcut and an intent cue. Before
training, fixed-budget replacements therefore make every operation/operand
cell exactly four A-first and four B-first examples (144/144 overall).

The binding label records whether canonical semantic argument A or B is
mentioned first. It supervises role order without asking the model to copy
arbitrary decimal strings. All three arms use this identical counterbalanced
corpus, tokenizer, number of updates, optimizer, pressure suite, and seeds.
The historical Phase-2 K=8/R178 cell is validated and retained as a secondary
reference only; it is not the matched gate baseline. The revised screen is 15
new runs rather than 10.

## Scores and gate

Report operation accuracy on every robustness track, macro and worst-group
operation accuracy, binding accuracy, and joint operation-plus-binding
accuracy. Keep classification protocol validity separate and trivially 100%
by construction.

The screen may advance only if a discriminative arm improves worst-group
operation accuracy over the paired generative baseline by at least 10
percentage points, its paired-bootstrap 95% lower bound is strictly positive,
and no operation-robustness track regresses by more than five points. A passing
arm must be confirmed on a newly frozen sealed suite before taxonomy growth.

The operation-plus-binding arm is preferred over operation-only only if its
operation result is non-inferior (no track more than five points worse) and it
improves joint binding utility. The development suite selects at most one arm.

## Execution guard

Preparation writes registration, fingerprints, and split manifests only.
Training requires an explicit `screen` stage:

```powershell
./.venv/Scripts/python.exe -m eval.semantic_bottleneck `
  --config configs/phase4_bottleneck_screen.yaml --stage prepare

./.venv/Scripts/python.exe -m eval.semantic_bottleneck `
  --config configs/phase4_bottleneck_screen.yaml --stage screen
```

## Results

| Arm | Mean worst-group operation | Paired worst delta | Paired macro delta | Gate |
| --- | ---: | ---: | ---: | --- |
| Matched generative | 1.11% | — | — | Baseline |
| Discriminative operation | 13.33% | +12.22 pp [+2.22, +22.22] | +5.11 pp [+0.11, +10.11] | Fail |
| Operation + binding | 26.66% | **+25.55 pp** **[+12.22, +33.33]** | **+9.44 pp** **[+5.33, +16.00]** | Fail |

The operation-only head crossed the primary worst-group and positive-CI
thresholds, but held-out-template operation accuracy regressed 5.55 pp, just
beyond the five-point limit. The multitask arm produced the strongest robust
representation result in the project so far, but discourse-distractor
operation accuracy regressed 6.67 pp and therefore also failed.

Relative to operation-only, the binding head was operation-noninferior under
the registered five-point rule. Its joint macro gain was +4.56 pp, but the CI
[-3.89, +15.78] includes zero. Binding accuracy was strong on familiar forms
and several shift tracks, while minimal-contrast binding repeatedly collapsed
to 0%. No arm was selected and no sealed confirmation suite was created.

The 15 new runs consumed 3,746.89 aggregate training seconds (62.45 minutes).

## Decision

Do not weaken the gate after seeing the result. Preserve the multitask arm as
a strong lead, but perform a read-only case-level audit of its discourse
regression and minimal-contrast binding collapse before registering another
intervention. Any follow-up must use a new development/sealed split and may not
tune against a sealed suite.

Artifacts:

- `runs/phase4_bottleneck_screen_v1/screen_results.json`
- `runs/phase4_bottleneck_screen_v1/screen_analysis.json`
- `runs/phase4_bottleneck_screen_v1/registration.json`
- `runs/phase4_bottleneck_screen_v1/split_manifest.json`
