# Compact evidence bundle

This directory makes the repository's central conclusions inspectable without
committing model weights or complete generated corpora. It contains 127 files
(approximately 10.5 MiB) copied from the local historical `runs/` tree at the
2026-08-28 research pause.

The separate [`archive_manifest.csv`](archive_manifest.csv) inventories all
1,869 externally archived files (53,316,875,958 bytes) without adding those
large objects to Git.

[`evidence_manifest.csv`](evidence_manifest.csv) records each source path,
Git-tracked path, byte size, SHA-256 digest, scientific-status label, source
commit, and durable archive URL. Files below `records/` preserve the source
path after the original `runs/` prefix. The controlled status vocabulary is
defined in the [documentation index](../README.md#scientific-labels).

## Central evidence

- Parser sealed confirmation:
  [`analysis.json`](records/phase4_stagec_invariance_v1/sealed_confirmation/analysis.json)
  and [`case_predictions.json`](records/phase4_stagec_invariance_v1/sealed_confirmation/case_predictions.json)
- Repaired capacity comparison:
  [`phase5_capacity_comparison_repaired.json`](records/phase5_capacity_comparison_repaired.json)
- Historical comparison with provenance caveats:
  [`phase5_capacity_comparison_legacy.json`](records/phase5_capacity_comparison_legacy.json)
- Paired value-swap analyses:
  [135M](records/phase5_grounding_value_swap_v1/smollm2_135m_agent/paired_probe_analysis.json)
  and [360M](records/phase5_grounding_value_swap_v1/smollm2_360m_capacity/paired_probe_analysis.json)
- Repaired frozen-model ledgers:
  [135M](records/smollm2_135m_agent/frozen_audit_repaired_v1/step_3000/case_predictions.jsonl)
  and [360M](records/smollm2_360m_capacity/frozen_audit_repaired_v1/step_3000/case_predictions.jsonl)
- Corpus validity records:
  [`corpus_audit.json`](records/phase5_validity_replay/corpus_audit.json)
  and [`corpus_manifest.json`](records/phase5_validity_replay/corpus_manifest.json)
- Recorded training timings:
  [135M](records/smollm2_135m_agent/agent_model/training_metrics.json)
  and [360M](records/smollm2_360m_capacity/agent_model/training_metrics.json)

## Limits

This is a documentary subset, not a turnkey reproduction package. Some reports
refer to complete run directories, checkpoints, or generated corpora available
only in the [full archive](https://media.lbsa71.net/LCE/index.html). The bundle preserves
negative and invalidated evidence deliberately; inclusion does not upgrade an
artifact's scientific status.

The source tree predates the documentation reorganization and corresponds to
Git commit `4990ba4` plus ignored local run outputs. Read the
[validity audit](../history/phase5_validity_audit.md) before reusing any data or
checkpoint.
