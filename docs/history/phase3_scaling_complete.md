# Phase 3: Sequential Training-Token Curves

**Date:** 2026-08-26
**Status:** Complete; repetition axis closed

## Design

The corrected Phase-2 factorial supplied R22, R89, and R178 cells. Phase 3
added R356 and then conditionally R889 for K=1 and K=8, using the same five
paired seeds, byte tokenizer, train/evaluation operands, pressure suite, model,
optimizer, and proportional warmup. R889 ran only after K=8 passed the
registered R178-to-R356 macro continuation gate.

## Full curve

| Breadth | Exposure | Macro robust mean | Worst-group mean |
| --- | ---: | ---: | ---: |
| K=1 | R22 | 39.00% | 30.00% |
| K=1 | R89 | 37.11% | 30.00% |
| K=1 | R178 | 34.44% | 18.89% |
| K=1 | R356 | 36.00% | 20.00% |
| K=1 | R889 | 38.33% | 24.44% |
| K=8 | R22 | 43.11% | 15.55% |
| K=8 | R89 | 42.22% | 14.44% |
| K=8 | R178 | 46.11% | 27.78% |
| K=8 | R356 | 50.11% | 28.33% |
| K=8 | R889 | 49.78% | 26.66% |

The R178-to-R356 K=8 macro gain was real locally (+4.00 pp per doubling, CI
[+1.33, +6.89]) but did not persist into the tail.

## Registered terminal interval

R356-to-R889 spans log2(2.5) resource doublings. All values below are accuracy
points per doubling, paired by seed:

| Breadth | Metric | Slope | Paired 95% CI | Classification |
| --- | --- | ---: | ---: | --- |
| K=1 | Macro robust | +1.76 pp | [-0.67, +4.79] | Flat/inconclusive |
| K=1 | Worst group | +3.36 pp | [-5.04, +15.13] | Flat/inconclusive |
| K=8 | Macro robust | -0.25 pp | [-5.04, +4.54] | Flat/inconclusive |
| K=8 | Worst group | -1.26 pp | [-3.79, 0.00] | Flat/inconclusive |

At K=8, held-out and lexical tracks moved slightly upward, but the discourse
track moved -6.73 pp per doubling with a wide interval. No individual terminal
track supplies a reliable positive semantic-invariance signal. Seed-level
worst-group outcomes at R889 ranged from 0% to 33.33%, so the mean is not a
stable floor.

## Decision

Close repetition scaling. It can produce a local average improvement, but the
five-point curve does not show a sustained macro tangent and never produces a
positive worst-group confidence bound. More repetitions therefore remain a
poor candidate for repairing form-invariant intent. Phase 4 should test the
representation objective directly using the prepared binding-balanced matched
screen.

The ten R889 runs consumed 3,556.36 aggregate training seconds (59.27 minutes).

## Artifacts

- `runs/breadth_reinforcement_v2/results.json`
- `runs/phase3_scaling_stage_a_r356_v1/results.json`
- `runs/phase3_scaling_stage_b_r889_v1/results.json`
- `runs/phase3_scaling_stage_b_r889_v1/full_curve_analysis.json`
- `configs/phase3_scaling_stage_a.yaml`
- `configs/phase3_scaling_stage_b.yaml`
