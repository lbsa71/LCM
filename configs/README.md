# Configuration status

The project is paused. These YAML files are retained for provenance and
software reproduction; none authorizes a new training run. Before reuse,
confirm that the corpus, teacher replay, tokenized terminal-target coverage,
runtime proof enforcement, seeds, and suite composition satisfy the deferred
[kill-test integrity gate](../docs/kill_test.md).

## Software smoke test

- `smoke.yaml` — inexpensive pipeline exercise only; not scientific evidence.

## Experiment-record configurations

These correspond to the corrected or final version of a documented historical
experiment:

- `form_variation_v2.yaml`
- `breadth_reinforcement_v2.yaml`
- `phase25_representation.yaml`
- `phase26_contrast_coverage.yaml`
- `phase26_confirmation.yaml`
- `phase28_discourse_coverage.yaml`
- `phase29_discourse_augmentation.yaml`
- `phase210_capacity_interaction.yaml`
- `phase3_scaling_stage_a.yaml`
- `phase3_scaling_stage_b.yaml`
- `phase4_bottleneck_screen.yaml`
- `phase4_stageb_factorial.yaml`
- `phase4_stagec_invariance.yaml`
- `phase5_validity_replay.yaml`
- `smollm2_135m_agent.yaml`
- `smollm2_360m_capacity.yaml`

They reproduce historical intent, not necessarily a scientifically valid run
under current standards. Consult the corresponding report in
[`docs/history/`](../docs/history/README.md).

## Exploratory benchmark

- `architecture_benchmark.yaml` — shared development pressure test. It includes
  superseded or missing historical inputs and must not be treated as a sealed
  architecture ranking.

## Superseded or diagnostic presets

- `form_variation.yaml` and `breadth_reinforcement.yaml` — replaced by v2.
- `small.yaml`, `small_v2.yaml`, `small_v3.yaml`, `primary.yaml`, `stretch.yaml`,
  and `decisive_3h.yaml` — early POC/scaling presets.
- `scaled_cot_300m.yaml`, `scaled_hybrid_300m.yaml`,
  `scaled_overnight_100m.yaml`, and `scaled_overnight_300m.yaml` — historical
  scratch runs whose corpora or interfaces do not support current architecture
  comparisons.
- `smollm2_360m_agent.yaml` — older 360M path; the capacity configuration above
  records the later run.

Do not select a configuration merely because its filename sounds canonical.
The repository [README](../README.md), not a YAML preset, is the present source
of truth.
