# Phase 4 Stage-B Failure Audit

**Date:** 2026-08-26
**Status:** Complete; read-only analysis

The Stage-B screen persisted 2,700 case predictions: 108 fresh development
cases × five arms × five paired seeds. The strongest arm was binding weight
0.25 with scaffold counterbalancing. Across the five robust tracks it reached
65.11% operation, 74.89% binding, and 50.44% joint accuracy.

Its improvement is structurally meaningful but incomplete. Minimal-contrast
operation accuracy rose to 88.89%, and complete three-operation consistency
for a shared scaffold rose from 0/30 groups in the generative baseline to
20/30. The remaining operation floor moved to lexical shift (35.56%) and
syntax reversal (53.33%). Across robust cases, per-operation accuracy was
72.67% ADD, 58.00% SUBTRACT, and 64.67% COMPARE. Binding accuracy was balanced
across A-first and B-first cases (73.78% and 76.00%).

The worst track varies by seed: lexical shift is the floor for seeds 29, 53,
and 67; syntax reversal is the floor for seeds 17 and 41. This is not a single
template hole suitable for another replacement curriculum.

The frozen-probe audit contains 2,160 records across four bottleneck arms. On
the strongest arm, a refitted closed-form operation probe is slightly worse
than the trained head overall (63.78% versus 65.11%), on lexical shift (34.44%
versus 35.56%), and on syntax reversal (51.11% versus 53.33%). It repairs zero
head errors. The response-boundary state therefore lacks a stable linearly
recoverable intent distinction on the limiting tracks.

This closes final-head/readout replacement as the immediate hypothesis. The
next efficient experiment directly regularizes same-intent representations
across opposing surface orders while preserving separate binding supervision.

Artifacts:

- [Case predictions](../evidence/records/phase4_stageb_factorial_v1/case_predictions.json)
- [Frozen-probe records](../evidence/records/phase4_stageb_factorial_v1/frozen_probe_records.json)
- [Frozen-probe summary](../evidence/records/phase4_stageb_factorial_v1/frozen_probe_audit.json)
