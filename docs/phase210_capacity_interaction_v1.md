# Phase 2.10 Capacity Interaction

**Date:** 2026-08-26
**Status:** Complete; wider capacity did not resolve the robustness trade-off.

## Design

Phase 2.10 reused the validated hidden-size-96 baseline and fixed-update
augmentation cells from Phase 2.9. It trained fresh paired baseline and
augmentation cells at hidden size 144 (intermediate size 576), keeping three
layers, four heads, fixed byte tokenizer, K=8 data, 1,600 updates, optimizer,
development suite, and five seeds fixed. No sealed suite was created or opened.

## Result

| Effect | Worst-group delta | 95% bootstrap CI | Macro delta | 95% bootstrap CI |
| --- | ---: | ---: | ---: | ---: |
| Width increase, baseline | +3.3 pp | [-15.6, +22.2] | +0.5 pp | [-6.0, +7.3] |
| Width increase, augmentation | -1.1 pp | [-16.1, +15.0] | +2.3 pp | [-6.2, +10.7] |
| Width x augmentation interaction | -4.5 pp | [-32.8, +32.2] | +1.9 pp | [-7.7, +14.7] |

Within the wide model, augmentation changed worst-group accuracy by -10.0 pp
(CI [-26.1, +10.6]), regressed held-out templates by 16.7 pp, and improved
discourse by an uncertain 11.1 pp. It failed the registered continuation gate.

## Decision

The tested width increase neither raises the general robustness floor nor
selectively helps the discourse augmentation. Combined with Phases 2.8-2.9,
this retires the fixed-template curriculum branch: deletion, extra exposure,
and modest width all fail to remove the specialization trade-off.

Return to the roadmap's scaling question using a sequential tail probe. The
existing corrected factorial already covers R22, R89, and R178 with no reliable
reinforcement slope. Add R356 at K=1 and K=8; continue toward R889 only if the
new per-doubling macro slope has a positive lower confidence bound without a
material worst-group regression.

## Artifacts

- [Registration](../runs/phase210_capacity_interaction_v1/registration.json)
- [Split manifests](../runs/phase210_capacity_interaction_v1/split_manifests.json)
- [Raw results](../runs/phase210_capacity_interaction_v1/screen_results.json)
- [Capacity analysis](../runs/phase210_capacity_interaction_v1/screen_analysis.json)
- [Configuration](../configs/phase210_capacity_interaction.yaml)
