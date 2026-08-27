# Phase 5: Pretrained-Agent Capacity Discriminator

**Date:** 2026-08-27
**Status:** Legacy, repaired-corpus, and value-swap probes complete; see the feasibility checkpoint

## Evidence entering the experiment

SmolLM2-135M ReAct reached 0% grounded success at SFT microstep 1,000, 43%
at 2,000, and 45% at 3,000. Its terminal gain is only +2 percentage points.
At the final checkpoint, retrieval/computation, missing-evidence, and
counterfactual suites remain at 0%; tool recovery is 10%. It fails the
registered overall, breadth, and terminal-slope criteria.

The compact two-layer semantic parser is also rejected at its tested scale: a
development-selected consistency arm failed sealed confirmation and made
held-out-template accuracy substantially worse. The legacy scratch-300M run is
diagnostic-only because its corpus is invalid for a family-level comparison.

## Registered question

Does increasing the pretrained backbone from SmolLM2-135M to SmolLM2-360M
materially raise broad grounded-agent competence under the exact same corpus,
SFT exposure, optimizer schedule, runtime, and held-out tasks?

The new run changes only `pretrained_model_name` and output location. Its five
data files are byte-for-byte copies of the 135M run; their SHA-256 hashes are
recorded in `runs/smollm2_360m_capacity/capacity_registration.json`. Both runs
use 3,000 microsteps, microbatch 4, accumulation 8, 375 optimizer updates,
12,000 sequences, and 12,288,000 padded tokens. The historical 135M trainer
did not enforce the configured seed; its actual seed is unknown. The 360M
trainer explicitly seeds with 42. Therefore this is an exposure-matched,
single-run comparison, **not a strictly capacity-only causal estimate**.

Both runs retain the historical scheduler behavior: the configured 200-step
warmup counts optimizer updates, whereas checkpoint steps count microbatches.
The 3,000-step scheduler is advanced only 375 times. No scheduler correction
was introduced during this capacity comparison.

### Evaluation and timing audit — 2026-08-27

The 360M training finished at 15:38, taking 30,601.89 seconds (8.50 hours),
10,200.63 ms per microstep and 401.54 padded tokens/s. The saved 135M time is
1,578.25 seconds (26.30 minutes): 19.39 times shorter. This is observed
wall-clock cost, not a hardware-controlled scaling law; the cause of the
large slowdown has not been established.

The old milestone evaluator ignored `agent_runtime`: it used 12 turns rather
than the configured 14. The shell also caps generated messages at 64 tokens
regardless of either the default 128 or configured 256-token budget. A new
versioned frozen-checkpoint replay explicitly records these effective limits,
hashes inputs/weights/source, and retains every episode and outcome. Use the
`legacy` runtime profile first to calibrate against historical scores. Keep
historical scores untouched and distinguish runtime ablations if run later.

## Decision rule

Evaluate steps 1,000, 2,000, and 3,000 on the identical 100 held-out tasks.
For pragmatic general-purpose readiness, require all three:

1. at least 70% overall grounded success at step 3,000;
2. at least 40% on each retrieval/computation, missing-evidence, recovery, and
   counterfactual core suite;
3. at least +5 pp overall grounded gain from step 2,000 to 3,000.

Failure rejects this 360M configuration as usable or fast-converging. Compare
its per-suite gains with 135M to decide whether pretrained capacity has a
credible slope. Only a meaningful cross-capacity gain justifies testing a
larger pretrained model; a flat result shifts the next experiment to
trajectory/objective design. It does not justify another unmodified scratch
run.

**Validity amendment:** the audit below and [corpus-validity report](phase5_validity_audit.md)
show that the original readiness gate contains an unanswerable counterfactual
suite and omitted multi-hop tasks. Retain the registered rule for historical
reference, but do not use its failure to reject the architecture family. A
repaired probe is development evidence, not a retroactive replacement for the
original registration.

## Frozen legacy replay result

The matched 100-task final-checkpoint replay reproduced 45% grounded success
for 135M and found 41% for 360M. Full-proof/observed-evidence rescoring gives the
same totals. The paired change is -4 pp; a world-cluster bootstrap gives
[-5.26, -1.92] pp **conditional on these two fixed runs and ten sampled worlds**.
This is not a seed-level confidence interval or a causal capacity estimate.

Both models are perfect on the sampled language, invariants, and direct-math
tracks. Single-hop grounded success is 40% versus 10%; recovery is 10% versus
0%. Both score zero on retrieval/computation and the defective abstention and
counterfactual tracks. The larger run has not bought useful breadth in this
diagnostic. Remaining legacy milestone evaluations were stopped; its partial
step-2000 ledger remains available, but is not scored as complete.

Artifacts: `runs/phase5_capacity_comparison_legacy.json` and each model's
`frozen_audit_legacy_v1/step_3000` directory. The original replay source is
preserved in `runs/phase5_legacy_source_20260827`. Its provenance amendment
corrects an assumed float32 manifest field: both loaded models actually use
bfloat16. Subsequent manifests measure dtype after loading.

The saved step CSVs show a 360M median of about 10.24 seconds both on ordinary
microsteps and optimizer-update microsteps, versus 0.432/0.457 seconds for
135M. The slowdown is therefore not confined to optimizer-update steps; this
does not identify its hardware cause. There were only 375 optimizer updates
and 12,000 sequences (6.52% of the 184,000-trajectory corpus) in either run.

## Execution

```powershell
.\.venv\Scripts\python.exe -m training.agent_sft `
  --config configs/smollm2_360m_capacity.yaml

.\.venv\Scripts\python.exe -m eval.eval_milestones `
  --config configs/smollm2_360m_capacity.yaml --per-suite 10
```

## Repaired-corpus comparison

The two frozen final checkpoints were evaluated sequentially on the same
110 cases from a fresh, 20-world development corpus. All eleven suites are
present, all proof references exist, and evidence-disabled/closed-book
controls use empty evidence views. Numerical runtime limits remain matched at
12 turns and an effective 64 generated tokens per message.

| Strict grounded success | 135M | 360M |
| --- | ---: | ---: |
| Overall | 53/110 (48.18%) | 55/110 (50.00%) |
| Single-hop | 20% | 10% |
| Multi-hop | 0% | 0% |
| Retrieval + computation | 10% | 10% |
| Missing evidence | 0% | 0% |
| Recovery | 10% | 40% |
| Direct computation | 100% | 100% |
| Counterfactual evidence | 90% | 90% |
| Evidence-disabled abstention | 0% | 0% |

Both also score 100% on the sampled language, invariant, and closed-book
tracks, which include repeated training forms. The capacity difference is
+1.82 pp, with a world-cluster 95% interval of [-2.20, +5.00] pp. It does not
establish a reliable overall gain. The sign differs from the old benchmark's
-4 pp result: this reinforces why the invalid benchmark must not decide the
architecture ranking. The corpus differs between old and repaired probes, so
their score change is not a controlled estimate of repair alone.

The larger checkpoint improves the sampled recovery track but has not earned
its 19.39-fold reported training-time cost. Neither checkpoint demonstrates
the breadth needed for a general-purpose agent. The next informative test is
evidence-value swapping, followed by a small, fully replay-validated trajectory
control—not another unmodified capacity increase. A repaired learning curve
across checkpoints and independent training seeds remains outstanding.

Artifacts: `runs/phase5_capacity_comparison_repaired.json` and each model's
`frozen_audit_repaired_v1/step_3000` directory. No saved weights were changed.

The paired new-value probe subsequently passed 8/10 complete pairs for 135M
and 9/10 for 360M (17/20 and 18/20 individual cases). Neither model reused the
old counterfactual or real-world target answers. See the
[feasibility checkpoint](phase5_feasibility_verdict.md) for the current opinion,
scope limitations, and next experiment.
