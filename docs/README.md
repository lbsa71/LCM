# LCM research record

The project is paused. The repository root [README](../README.md) is the sole
current synthesis of the research question, evidence, limitations, and
conclusion. Nothing in this directory authorizes a training run.

## Documentation authority

1. [README](../README.md) — paper-style synthesis and present status.
2. [Kill-test protocol](kill_test.md) — deferred conditions for resumption.
3. [Artifact and provenance policy](artifacts.md) — where evidence is retained.
4. [Environment snapshot](environment.md) — pause-time software and hardware.
5. [Historical experiment ledger](history/README.md) — dated reports, including
   negative, invalidated, exploratory, sealed, and unrun work.

The historical reports preserve the sequence in which decisions were made.
Their old “next” or “active” language is not current project direction.

## Principal findings

| Finding | Evidence | Status |
| --- | --- | --- |
| Familiar forms were learned, but the compact parser did not develop robust operation invariance. | [Form variation](history/form_variation_clean_v2.md), [breadth/reinforcement](history/breadth_reinforcement_v2.md), [complete token curve](history/phase3_scaling_complete.md) | Multi-seed negative result |
| The selected consistency candidate improved agreement, not reliable correctness, and failed sealed confirmation. | [Stage C invariance](history/phase4_stagec_invariance.md) | Sealed rejection |
| Pretrained 135M and 360M policies followed changed evidence on a small paired diagnostic. | [Feasibility verdict](history/phase5_feasibility_verdict.md) | Narrow development evidence |
| Increasing capacity from 135M to 360M produced no reliable aggregate gain and cost substantially more recorded training time. | [Capacity report](history/phase5_pretrained_capacity.md) | Inconclusive capacity comparison |
| Historical corpus and benchmark defects materially limit earlier conclusions. | [Validity audit](history/phase5_validity_audit.md) | Authoritative validity warning |
| The broad synthetic-only, fact-externalized scratch-model thesis remains untested. | [README](../README.md) | Current conclusion |

## Repository map

- `agent/`, `synth/`, `training/`, and `eval/` contain the research prototype.
- `tests/` verifies software and protocol invariants; it does not certify the
  scientific validity of generated corpora or reported experiments.
- `configs/` preserves historical presets. See [the configuration index](../configs/README.md)
  before using any of them.
- `docs/evidence/` contains a small, Git-tracked documentary subset of metrics,
  registrations, manifests, and case-level results.
- Full corpora, ledgers, and checkpoints are intentionally outside Git and are
  archived at [media.lbsa71.net/LCE](https://media.lbsa71.net/LCE/index.html).

## Scientific labels

The machine-readable evidence manifest uses the following controlled labels:

- **sealed** — evaluated once after configuration selection;
- **development** — useful for diagnosis, not a final confirmation;
- **exploratory** — architecture or failure analysis without a confirmatory
  claim;
- **superseded** — replaced by a later corrected experiment;
- **historical** — retained provenance whose original interpretation must be
  read through the later validity audit;
- **diagnostic-invalidated** — useful failure evidence whose defect prevents
  the original score from supporting its intended inference; and
- **validity-audit** — evidence produced specifically to identify or quantify
  defects in an earlier corpus, run, or evaluation.

Narrative reports may additionally use **invalidated** as the plain-language
form of `diagnostic-invalidated`, and **unrun** for proposals with no registered
or launched training. Every row in `docs/evidence/evidence_manifest.csv` uses
one of the seven controlled labels above.

Negative and invalidated results are retained because they constrain future
work and document failure modes in synthetic-data research.
