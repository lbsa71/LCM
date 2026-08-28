# Clean Form-Variation Curve (Phase 0–1)

> **Archived experiment record.** “Active roadmap” language below describes
> the 2026-08-25 research state and is no longer current. See the repository
> [README](../../README.md) for the paused project status.

**Date:** 2026-08-25  
**Status:** Complete. This is the first clean result in the active
[research roadmap](research_roadmap.md).

## Protocol

Twenty scratch-parser runs were executed: K = 1, 2, 4, 8 expression forms per
operation, with five paired seeds at every K. Every run had 400 full-batch
updates and 819,200 padded input tokens. The tokenizer was fit only on that
condition's training examples.

The fixed external suite contains 126 fact-free examples in six groups. Its
utterances are absent from every training and tokenizer corpus; the per-K
split manifests record 0 exact overlaps and `PASS` validation in
`runs/form_variation_v2/split_manifests.json`.

The primary result is the **worst robust group** per seed, taken across held
templates, lexical shifts, syntax/order reversal, discourse distractors, and
minimal contrasts. Macro robust accuracy is reported as a secondary diagnostic.

## Results

| K forms/intent | Familiar form, new operands | Worst robust group | Macro robust | Held templates | Lexical | Syntax/order | Discourse | Minimal contrast |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 100.0% | 33.3% | 33.8% | 35.6% | 33.3% | 33.3% | 33.3% | 33.3% |
| 2 | 100.0% | 27.2% | 36.1% | 27.2% | 33.3% | 40.0% | 40.0% | 40.0% |
| 4 | 100.0% | 23.3% | 46.3% | 36.7% | 40.0% | 61.1% | 44.4% | 48.9% |
| 8 | 100.0% | 25.0% | 41.9% | 28.9% | 36.7% | 47.8% | 57.8% | 37.8% |

The model learns all familiar language forms, including new numbers, at every
K. The failure is therefore form generalization rather than arithmetic or
number-token generalization.

### Local slopes, paired by seed

| Doubling | Worst robust slope (pp/doubling, 95% bootstrap CI) | Macro robust slope (pp/doubling, 95% bootstrap CI) |
| --- | ---: | ---: |
| K=1 → 2 | −6.1 [−15.0, 0.0] | +2.3 [−1.7, +9.0] |
| K=2 → 4 | −3.9 [−8.3, 0.0] | +10.2 [+2.1, +18.3] |
| K=4 → 8 | +1.7 [−7.8, +15.6] | −4.4 [−12.7, +4.2] |

## Interpretation and decision

Increasing wording diversity gives a real **average** improvement through K=4,
especially in syntax/order handling. But it does not lift the worst failure
mode above the three-way chance floor, and the K=8 result regresses on the
macro score. The architecture has not learned an operation-invariant semantic
boundary; it has learned partially overlapping surface clusters.

Under the roadmap’s primary worst-group policy, there is no evidence that more
K alone is the productive next scale axis. The first two adjacent intervals
already have an upper confidence bound below +1 point per doubling. The noisy
K=4→8 interval means this is a **provisional saturation finding**, not proof
of an absolute ceiling.

The next registered experiment is therefore Phase 2: hold total tokens fixed
while separately varying template breadth and repetitions per template. That
will tell us whether K=4's average improvement comes from coverage or from an
uncontrolled change in examples-per-template.

## Artifacts and reproduction

```powershell
.\.venv\Scripts\python.exe -m eval.form_variation --config configs/form_variation_v2.yaml
```

- `runs/form_variation_v2/results.json` — seed-level measurements.
- `runs/form_variation_v2/analysis.json` — paired slopes and confidence intervals.
- `runs/form_variation_v2/split_manifests.json` — holdout validation.
