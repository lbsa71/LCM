# Phase 2.9 Replacement versus Augmentation

**Date:** 2026-08-26
**Status:** Complete; neither augmentation arm passed the registered gate.

## Design

Phase 2.9 reused the validated K=8/R89 baseline and discourse-replacement cells
from Phase 2.8. Two new five-seed cells retained all 288 standard examples and
appended 72 counterbalanced discourse examples. One used the original 1,600
updates; the other used 2,000 updates to preserve R89 exposure over the larger
360-example corpus. No sealed suite was created or opened.

## Result versus baseline

| Cell | Worst-group delta | Held-out templates | Syntax order | Discourse |
| --- | ---: | ---: | ---: | ---: |
| Replacement (reused) | -7.2 pp | -17.2 pp | -20.0 pp | +20.0 pp |
| Augmentation, 1,600 updates | -5.6 pp | -21.7 pp | -15.6 pp | +7.8 pp |
| Augmentation, 2,000 updates | -17.8 pp | -16.7 pp | -25.6 pp | +24.5 pp |

Neither new cell passed the +10 pp worst-group/positive-CI/no-regression gate.
The paired effect of increasing augmentation training from 1,600 to 2,000
updates was -12.2 pp worst-group accuracy (95% bootstrap CI [-18.9, -4.5]);
the macro effect was -3.2 pp (CI [-8.1, +1.1]).

## Interpretation and decision

Deleting standard examples is not the main cause of the robustness trade-off:
augmentation retained every standard form yet still regressed held-out and
syntax tracks. Insufficient exposure is also not the cause; restoring exposure
made the worst-group result reliably worse while increasing discourse gains.
Optimization is strengthening a specialized cue policy within the current
model. Do not spend further runs on repetition or corpus-budget variants.

The next efficient test is a capacity interaction screen: reuse the 96-wide
baseline and fixed-update augmentation, then train the same pair at a wider
model size. This directly tests whether capacity permits coexistence of the
standard and discourse policies rather than merely shifting the boundary.

## Artifacts

- [Registration](../runs/phase29_discourse_augmentation_v1/registration.json)
- [Split manifests](../runs/phase29_discourse_augmentation_v1/split_manifests.json)
- [Raw results](../runs/phase29_discourse_augmentation_v1/screen_results.json)
- [Gate analysis](../runs/phase29_discourse_augmentation_v1/screen_analysis.json)
- [Configuration](../configs/phase29_discourse_augmentation.yaml)
