# Breadth versus Reinforcement Factorial (Phase 2)

**Date:** 2026-08-25  
**Status:** Complete; corrected v2 is the scientifically usable run.

## Design

The factorial crosses four surface-form breadth levels (K = 1, 2, 4, 8
templates per operation) with three exposure levels (R ≈ 22, 89, 178
presentations per unique training example). Every cell has five paired seeds:
60 runs total.

All cells use the same fixed byte tokenizer, model size, optimizer and 126-case
pressure suite. Warmup is scaled to 10% of each cell's update budget. The
per-K manifests report zero exact train/test utterance overlap.

The earlier `runs/breadth_reinforcement_v1` run is preserved as a diagnostic
artifact, but is not used here: it held warmup at 40 steps, confounding cells
with very different update budgets.

## Cell means

| K | R | Worst robust group | Macro robust accuracy |
|---:|---:|---:|---:|
| 1 | 22 | 30.0% | 39.0% |
| 1 | 89 | 30.0% | 37.0% |
| 1 | 178 | 18.9% | 34.4% |
| 2 | 22 | 3.3% | 23.0% |
| 2 | 89 | 3.3% | 27.0% |
| 2 | 178 | 3.3% | 25.0% |
| 4 | 22 | 26.7% | 38.0% |
| 4 | 89 | 20.0% | 39.0% |
| 4 | 178 | 26.7% | 43.0% |
| 8 | 22 | 16.0% | 43.0% |
| 8 | 89 | 14.0% | 42.0% |
| 8 | 178 | 28.0% | 46.0% |

Familiar forms with unseen operands remain at 100% in every cell. The failure
is therefore not arithmetic or number-token learning; it is robustness to
surface form and discourse structure.

## Slopes

Macro robust breadth slopes (percentage points per doubling of K):

- K=1 → 2: negative, −15.9 pp at R=22, −10.6 pp at R=89, −9.9 pp at R=178;
- K=2 → 4: positive, +15.3 pp, +12.8 pp, and +18.6 pp respectively; all
  paired bootstrap intervals exclude zero;
- K=4 → 8: small and uncertain, +4.7 pp, +2.9 pp, and +3.0 pp; all intervals
  include zero.

Reinforcement slopes are small and uncertain at every K. For example, the
R=89 → R=178 macro slopes are −2.7 pp (K=1), −2.0 pp (K=2), +3.8 pp (K=4),
and +3.9 pp (K=8), with all 95% intervals including zero.

## Interpretation

There is evidence for a useful breadth transition from K=2 to K=4 in average
robustness. There is no evidence that simply repeating the same forms more
often improves robustness. Additional breadth beyond K=4 is not yet a
statistically reliable gain.

The primary worst-group metric remains near chance and is highly variable. The
parser has learned a mixture of surface clusters, not a stable intent boundary.
The practical next step is therefore not more repetitions or a larger K sweep;
it is a representation/coverage intervention: argument binding, explicit
minimal-contrast supervision, and a sealed confirmation suite for the K=4/K=8
candidate configurations.

## Artifacts

- [Raw results](../runs/breadth_reinforcement_v2/results.json)
- [Paired slopes and confidence intervals](../runs/breadth_reinforcement_v2/analysis.json)
- [Split manifests](../runs/breadth_reinforcement_v2/split_manifests.json)
- [Reproduction config](../configs/breadth_reinforcement_v2.yaml)
