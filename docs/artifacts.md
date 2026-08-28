# Research artifacts and provenance

## Durable locations

- **Browsable archive:** [https://media.lbsa71.net/LCE/index.html](https://media.lbsa71.net/LCE/index.html)
- **Backing share:** `\\192.168.1.4\usb\Projects\s3-backup\media.lbsa71.net\LCE`
- **Git-tracked evidence subset:** [`docs/evidence/`](evidence/README.md)
- **Local working artifacts:** ignored `runs/` and `scratch/` directories

The network archive is the durable home for full checkpoints, generated
corpora, ledgers, and historical run trees. Git contains only compact evidence
needed to inspect the paper's conclusions without downloading model weights.

## Retention policy

Retain:

- registrations, configurations, seeds, split and corpus manifests;
- aggregate metrics and timing records;
- sealed predictions, paired analyses, and failure taxonomies;
- the exact final SmolLM2-135M and 360M adapters needed for a later kill test;
- negative and invalidated artifacts with explicit validity labels; and
- historical scratch checkpoints as diagnostics, not comparative evidence.

Do not place ordinary model weights, complete corpora, access tokens, local
environment files, or generated pipeline state in Git.

## Interpretation

Artifacts preserve what was run; they do not make every historical result
valid. In particular:

- the old scratch-300M corpus and interface do not support a ranking against
  current pretrained models;
- historical pretrained-agent data contained missing documents, unfinished
  trajectories, and absent suite coverage;
- the 135M historical seed was not enforced; and
- repaired development probes are not sealed confirmations.

See the [validity audit](history/phase5_validity_audit.md) before reusing any
training data or checkpoint.

## Archive verification

The archive is copied non-destructively. Local sources are retained. The
[machine-readable archive manifest](evidence/archive_manifest.csv) records
1,869 files, 53,316,875,958 bytes, source SHA-256 digests, and public URLs for
the complete `runs/` and `scratch/` trees. A post-transfer audit checked every
manifest path and byte length: all 1,869 files matched, with zero missing or
mismatched entries. A deterministic 18-file SHA-256 sample also matched in
full, including the exact final SmolLM2-135M and SmolLM2-360M step-3000
adapters. The manifest itself is published at the archive root.

Paths in historical reports that begin with `runs/` refer to the corresponding
path below the archive root. For example, `runs/example/result.json` maps to
`https://media.lbsa71.net/LCE/runs/example/result.json`.

Credentials are deliberately excluded. The repository previously tracked
local environment material; current policy treats any historical values as
exposed and requires rotation before reuse.
