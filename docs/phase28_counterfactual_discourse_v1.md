# Phase 2.8 Counterfactual Discourse Coverage

**Date:** 2026-08-26
**Status:** Complete; failed the registered continuation gate.

## Design

At K=8/R89, five fresh paired seeds compared a standard baseline with a
fixed-budget counterfactual-discourse curriculum. The intervention replaced
25% of examples in every operand/operation cell. Distractor cue families were
exactly counterbalanced across `ADD`, `SUBTRACT`, and `COMPARE`, while model,
tokenizer, optimizer, 1,600-update schedule, and total examples were fixed.
The existing development pressure suite was used for screening. No sealed
suite was created or opened.

## Result

| Outcome | Paired delta | 95% bootstrap CI |
| --- | ---: | ---: |
| Worst robust group | -7.2 pp | [-20.6, 0.0] |
| Held-out templates | -17.2 pp | [-33.9, -0.6] |
| Lexical shift | -6.7 pp | [-20.0, 0.0] |
| Syntax-order reversal | -20.0 pp | [-30.0, -4.4] |
| Discourse distractor | +20.0 pp | [+7.8, +32.2] |
| Minimal contrast | 0.0 pp | [-20.0, +20.0] |

The intervention learned the intended discourse distinction, but did not
produce a generally stronger parser. It moved probability mass from held-out
form and syntax robustness into discourse robustness. The registered gate
required +10 pp worst-group gain with a positive lower confidence bound and
no track regression over 5 pp; the arm failed all relevant conditions.

## Decision

Do not advance to sealed confirmation. The next experiment should distinguish
whether the trade-off is caused by deleting standard forms to keep the example
budget fixed, or by limited model/optimization capacity. Reuse this screen's
validated baseline and replacement arms, then compare discourse augmentation
at fixed updates with augmentation at matched per-example exposure.

## Artifacts

- [Registration](../runs/phase28_counterfactual_discourse_v1/registration.json)
- [Split manifests](../runs/phase28_counterfactual_discourse_v1/split_manifests.json)
- [Raw results](../runs/phase28_counterfactual_discourse_v1/screen_results.json)
- [Gate analysis](../runs/phase28_counterfactual_discourse_v1/screen_analysis.json)
- [Configuration](../configs/phase28_discourse_coverage.yaml)
