# Phase 2.5 Representation Diagnostic and Ablation

**Date:** 2026-08-26  
**Status:** Complete; no arm passed the preregistered continuation gate.

## Execution

Stage A evaluated the ten existing corrected-v2 checkpoints at K=4/R89 and
K=8/R89. It fit closed-form probes on frozen response-boundary states using
training forms only; the existing 126-case pressure suite was treated as a
development suite.

Stage B trained three new K=4/R89 arms, each with five paired seeds and the
same 800-update/80-warmup schedule. The corrected-v2 baseline was reused after
all five source corpora, checkpoints, and schedules passed validation. The
three new arms were operation-only plus minimal contrasts, typed frames, and
typed frames plus minimal contrasts. No sealed confirmation suite was created
or opened.

## Stage A — frozen probes

| K | Worst-group probe accuracy | Macro probe accuracy | Canonical A/B role accuracy | Mean absolute value error |
|---:|---:|---:|---:|---:|
| 4 | 26.7% [13.3%, 33.3%] | 46.6% [38.1%, 52.5%] | 92.9% | 227.4 |
| 8 | 27.8% [21.1%, 33.3%] | 47.6% [42.0%, 53.8%] | 91.8% | 215.2 |

The frozen state contains useful approximate role-order information, but not a
stable held-out intent boundary. Minimal-contrast triplet consistency was 0%
in every Stage-A seed.

## Stage B — K=4 screen

| Arm | Worst robust mean | Macro robust mean | Worst-group paired delta |
| --- | ---: | ---: | ---: |
| Reused operation-only baseline | 20.0% | 39.3% | — |
| Operation-only + minimal contrasts | 33.3% | 57.6% | +13.3 pp [0.0, +26.7] |
| Typed frame | 8.9% | 31.7% | −11.1 pp [−27.8, +5.6] |
| Typed frame + minimal contrasts | 24.4% | 42.2% | +4.4 pp [−17.8, +26.7] |

The continuation gate required at least +10 pp worst-group gain, a paired
bootstrap lower bound strictly above zero, and no robustness-track regression
over 5 pp. No arm passed:

- Minimal contrasts had the strongest average improvement, but its lower
  confidence bound was exactly zero.
- Typed frames regressed discourse distractors by 26.7 pp.
- Typed frames plus contrasts regressed held-out templates by 12.2 pp and
  discourse distractors by 18.9 pp.

Typed-frame minimal-contrast scores were 63.3% for argument binding and 45.6%
for full-frame exact match. Reported protocol validity is 100% because the
current evaluator chooses among constrained canonical candidates; it is not a
free-generation protocol result.

## Interpretation and decision

The screen gives a promising but inconclusive signal for minimal-contrast
supervision. The result is not strong enough to justify the sealed K=4/K=8
confirmation suite under the registered gate. The typed-frame target, as
implemented here, is not yet a useful drop-in replacement: it adds an output
burden without improving robust routing.

Do not start Phase 3 token scaling or repetition-only scaling. A seed-level
follow-up found that minimal-contrast training improved macro robustness in
all five seeds, while the worst-group floor usually moved to lexical shift.
The next registered action is therefore Phase 2.6: reuse the baseline and
minimal-contrast arms, add lexical-only and minimal-plus-lexical arms, and
measure their paired main and interaction effects before revisiting typed
frames or auxiliary binding losses.

## Artifacts

- [Stage-A probe results](../runs/phase25_representation_v1/stage_a_probe_results.json)
- [Stage-B raw results](../runs/phase25_representation_v1/stage_b_results.json)
- [Stage-B gate analysis](../runs/phase25_representation_v1/stage_b_analysis.json)
- [Registration](../runs/phase25_representation_v1/registration.json)
- [Reproduction config](../configs/phase25_representation.yaml)
