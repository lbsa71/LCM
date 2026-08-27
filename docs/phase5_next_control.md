# Phase 5 next control: complete, evidence-conditioned trajectories

## Why this is next

The repaired frozen-model comparison does not establish an overall capacity
gain worth the 360M run's reported cost. Both models still fail retrieval
abstention, and the historical corpus never supervised completion of those
trajectories. This is a direct, inexpensive hypothesis to test before more
capacity or scratch pretraining.

## Design to register before training

Use the 135M final checkpoint as the initial model for a small adaptation
screen. Reuse the generic model loader, SFT runner, evidence tools, and frozen
evaluation ledger; do not specialize or remove the other experiment classes.

- Generate matched questions in worlds with evidence present, evidence absent,
  and explicitly withheld evidence. Balance present/absent outcomes so that
  always abstaining cannot pass.
- Every teacher trajectory must complete and replay through the real tools.
  Reject fabricated search hits, fabricated observations, oracle-answer plans,
  missing proof lines, and final targets lost to token truncation.
- Freeze a development set with unseen entities and independently varied
  values before training. Preserve a separate, unopened confirmation set with
  held-out phrasing. Do not reuse the current diagnostic probe as confirmation.
- Compare a continued-training control against the complete-abstention arm at
  the same optimizer-update and sequence budget, with five paired new seeds.
  Record actual supervised/non-padding tokens as well as padded tokens.
- Keep model size fixed. Name microsteps and optimizer updates separately and
  explicitly register scheduler units; do not inherit the ambiguous old step
  convention silently.
- Primary endpoint: worst-group strict grounded success across present,
  absent, and withheld evidence. Require a positive paired improvement and no
  meaningful regression on evidence-present or direct-computation controls.
  Specify numerical thresholds and the small compute cap before launch.

## Status

Design direction only: this is **not** a registered training run, and no new
training has started. The remaining teacher-replay and tokenized-target
coverage checks must pass first. A successful screen would justify a broader
learning curve, not a declaration that general-purpose agency is solved.
