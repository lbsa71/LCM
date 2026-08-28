# Deferred LCM kill-test protocol

**Status:** proposed and paused on 2026-08-28. Not registered. Not authorized
for training.

## Decision question

The test asks a practical question rather than attempting to disprove every
possible fact-externalized architecture:

> Does the LCM representation or runtime provide a material reliability,
> latency, energy, or privacy advantage over a contemporary pretrained small
> function-calling model on externally anchored tasks?

Failure ends custom model development. Passing permits a separately approved
scratch experiment; it does not establish general-purpose agency.

## Initial investment cap

- 14 working days;
- 120 engineering hours;
- approximately 50 GPU-hours on the existing local GPU; and
- no scratch pretraining during the capped system comparison.

Crossing any cap requires a new decision. The likely and acceptable outcome is
termination after the pretrained comparison.

## Gate 0: scientific integrity

Before generating training data or opening a sealed suite:

1. remove teacher-side target-document injection;
2. remove fallback operands, fallback answers, and fabricated observations;
3. replay every teacher trajectory through the real tools;
4. reject incomplete trajectories and missing proof lines;
5. verify after tokenization that every supervised terminal target survives;
6. require final citations to have been observed and to satisfy the proof
   graph or final-state contract;
7. make configured seeds, optimizer updates, scheduler units, supervised
   tokens, and non-padding tokens auditable; and
8. require every declared suite to be nonempty and counterbalanced.

The full unit and integration suite must pass after these changes. This gate is
necessary but not sufficient: a green software suite does not validate the
scientific design.

## Benchmark

Freeze development and sealed partitions before training. Combine independently
reviewed human language with deterministic, stateful, verifier-backed cases.
The suite must include:

- English and Swedish requests, paraphrases, typos, ellipsis, and discourse
  distractors;
- present, absent, explicitly withheld, conflicting, and changed evidence;
- paired value swaps and prior reversals;
- genuine multi-hop state changes and retrieval-plus-computation;
- ambiguity, clarification, and selective abstention;
- unauthorized-action and prompt-injection probes; and
- final environment-state scoring rather than answer-string matching alone.

Keep training, tokenizer, development, and sealed language disjoint. Report
world- or task-cluster uncertainty rather than treating numeric variations as
independent language examples.

## Compared systems

Run identical tools, budgets, and tasks for:

1. a contemporary pretrained small function-calling model;
2. SmolLM2-135M producing direct ReAct actions;
3. the same pretrained backbone producing typed executable dataflow;
4. a deterministic oracle; and
5. a stronger general model as a reference ceiling.

Use at least three paired training seeds for trainable arms and repeated
inference for reliability. A clean scratch arm is conditional on the typed
system first earning continuation.

## Metrics and stopping rules

Primary metrics are worst-group grounded final-state success, proof-validity
rate, selective accuracy, and repeated-trial reliability. Also report latency,
energy, monetary cost, escalation rate, and every core-group regression.

Freeze numerical gates before opening the sealed suite. The current candidate
gates are:

- at least +10 percentage points in worst-group success, or at least a 3×
  latency/energy advantage at matched reliability;
- no core group regression greater than five points;
- at least 90% correct behavior when evidence is absent or withheld;
- at least 70% multi-hop final-state success; and
- a clear Pareto advantage over the external small-model baseline.

If a central gate fails, stop. Do not respond by adding templates, repetition,
or modest width: those escape routes were already tested.

## Conditional scratch extension

Only a successful system-level comparison can justify this separate phase. A
rough 20-token-per-parameter sanity point implies about 2.7B tokens for 135M or
6B for 300M. At the historical 300M throughput, those are approximately 45–60
and 100+ GPU-hours per seed before SFT, evaluation, failures, and replication.
Corpus validity and linguistic coverage are likely more expensive than the
raw compute.

Before resuming, record the new external baseline, benchmark manifest, exact
resource cap, frozen gates, and explicit user authorization in this document.
