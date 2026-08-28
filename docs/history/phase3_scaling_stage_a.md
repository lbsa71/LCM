# Phase 3 Stage A: Sequential Training-Token Scaling

> **Superseded checkpoint.** The [complete Phase 3 report](phase3_scaling_complete.md)
> contains the full curve and current interpretation.

**Date:** 2026-08-26
**Status:** Complete; continuation gate passed

## Question

Does doubling per-example reinforcement beyond R178 still improve robust form
generalization, or has the Phase-2 curve become flat or harmful? Stage A reused
the five-seed K=1 and K=8 R178 cells and trained matched R356 cells. All data,
tokenizer, architecture, optimizer, evaluation suites, and seeds were held
fixed.

## Registered continuation gate

Continue to R889 only if at least one breadth has a macro-robustness gain per
doubling whose paired-bootstrap 95% lower bound is strictly above zero, while
its mean worst-group regression is no greater than five percentage points.

## Results

| Breadth | Metric | R178 mean | R356 mean | Paired delta | Paired 95% CI |
| --- | --- | ---: | ---: | ---: | ---: |
| K=1 | Macro robust | 34.44% | 36.00% | +1.56 pp | [-2.89, +5.11] |
| K=1 | Worst group | 18.89% | 20.00% | +1.11 pp | [-16.67, +20.00] |
| K=8 | Macro robust | 46.11% | 50.11% | **+4.00 pp** | **[+1.33, +6.89]** |
| K=8 | Worst group | 27.78% | 28.33% | +0.56 pp | [-5.00, +6.67] |

The K=8 macro result passes the positive-slope condition, and its worst-group
mean does not regress. K=1 remains inconclusive. At K=8, the largest track
change was lexical shift (+13.33 pp, CI [+2.22, +24.45]); syntax/order changed
-3.33 pp, while held-out templates, discourse distractors, and minimal
contrasts moved +3.33, +4.44, and +2.22 pp respectively.

The ten R356 runs consumed 1,866.62 aggregate training seconds (31.11 minutes).
The K=8 seed-level worst-group results remained variable, so the conclusion is
the paired macro slope—not evidence that the worst group itself has been
solved.

## Decision

The registered gate passes. Run the R889 tail for both K=1 and K=8 with the
same five paired seeds. This is a continuation test, not a revision of the
metric or curriculum. After R889, estimate R356-to-R889 slopes and the overall
curve before deciding whether repetition scaling has a credible ceiling trend.

Before opening R889 results, classify the terminal interval as follows. A
macro-robustness paired-bootstrap lower bound above zero, with no mean
worst-group regression beyond five points, is evidence that repetition still
improves average sample efficiency. It is not evidence that repetition repairs
semantic invariance unless the worst-group slope's own paired-bootstrap lower
bound is also above zero. An interval containing zero is inconclusive/flat;
a negative upper bound is harmful. R889 closes this scale axis regardless of
category—the experiment is intended to estimate the local tangent, not to
extend repetitions indefinitely—and Phase 4 follows next.

Artifacts:

- `runs/breadth_reinforcement_v2/results.json` (R178 source cells)
- `runs/phase3_scaling_stage_a_r356_v1/results.json` (R356 cells)
- `configs/phase3_scaling_stage_a.yaml`
- `configs/phase3_scaling_stage_b.yaml`
