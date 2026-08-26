# Phase 2.6 Targeted Contrast-Coverage Result

**Date:** 2026-08-26  
**Status:** Complete; screen signal rejected by independent sealed confirmation.

## Screen

At K=4/R89, the registered 2x2 screen reused the validated baseline and
minimal-contrast arms and trained five new seeds each for lexical-only and
minimal-plus-lexical coverage. The combined arm passed the screen gate: its
worst-group delta was +21.1 pp (95% paired bootstrap CI [+3.3, +38.9]) with
no development-track regression greater than 5 pp. Lexical-only did not pass.

The factorial estimates showed modest, non-decisive complementarity: the
worst-group interaction was +1.1 pp (CI [-11.1, +12.2]). This was a selection
signal, not confirmation.

## Sealed confirmation

The selected combined arm and a fresh baseline were retrained from scratch for
five new paired seeds at K=4 and K=8. K=4 used 800 updates and K=8 used 1,600,
maintaining R89 effective exposure. The newly created isolated pressure suite
was only evaluated after registration; no screen checkpoint or seed was reused.

| Breadth | Worst-group delta | Gate outcome | Material track result |
| --- | ---: | --- | --- |
| K=4 | +3.3 pp [-6.7, +16.7] | Fail | Syntax-order reversal -14.5 pp [-27.8, -1.1] |
| K=8 | -26.7 pp [-33.3, -17.8] | Fail | Discourse distractor -37.8 pp [-53.3, -20.0] |

The confirmation gate required at least +10 pp worst-group gain, a paired
bootstrap lower bound strictly above zero, and no individual robustness-track
regression over 5 pp at both breadths. Neither breadth passed.

## Decision

Do not promote minimal-plus-lexical contrast coverage as a robust curriculum,
and do not use its screen result as evidence of a scaling benefit. The results
instead demonstrate a distributional trade-off: the intervention can improve
lexical wording while displacing syntax or discourse robustness, increasingly
at K=8. The next efficient step is a no-training failure audit that identifies
the specific sealed cases and operation confusions behind those two regressions
before registering another curriculum intervention.

## Artifacts

- [Screen results](../runs/phase26_contrast_coverage_v1/screen_results.json)
- [Screen analysis](../runs/phase26_contrast_coverage_v1/screen_analysis.json)
- [Confirmation registration](../runs/phase26_contrast_coverage_v1/confirmation/registration.json)
- [Confirmation results](../runs/phase26_contrast_coverage_v1/confirmation/results.json)
- [Confirmation analysis](../runs/phase26_contrast_coverage_v1/confirmation/analysis.json)
- [Screen configuration](../configs/phase26_contrast_coverage.yaml)
- [Confirmation configuration](../configs/phase26_confirmation.yaml)
