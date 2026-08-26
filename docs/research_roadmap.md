# LCM Research Roadmap: Robust-Generalization Slopes

**Date:** 2026-08-26  
**Status:** Active roadmap; Phase 2.6 registered and awaiting training approval  
**Supersedes:** the implicit “continue a large from-scratch run” direction in
the prior architecture decision. It does not supersede the evidence recorded
there; it changes the order in which we spend training compute.

## Resume checkpoint — 2026-08-26

No experiment is currently running. The completed artifacts are preserved and
Phase 2.5 is complete. Phase 2.6 has been implemented, tested, registered, and
prepared, but its ten new training runs have not started. Both reused arms and
all four split manifests pass validation. The sealed confirmation suite
remains untouched.

| Phase | State | Resume note |
| --- | --- | --- |
| 0 — Clean semantic benchmark | Complete | Fixed 126-case pressure suite; all split manifests pass with zero exact overlap. |
| 1 — Form-diversity curve | Complete | Twenty five-seed runs; familiar forms mastered, worst-group robustness remains near chance. |
| 2 — Breadth versus reinforcement | Complete | Corrected 60-cell v2 factorial is the scientific result; repetition has no reliable slope. |
| 2.5 — Representation diagnostic and ablation | **Complete; gate not passed** | Ten frozen probes and 15 new K=4/R89 runs completed. Minimal contrasts improved the mean but its confidence bound touched zero; typed targets regressed distractor tracks. |
| 2.6 — Targeted contrast coverage | **Registered; not run** | Baseline and minimal arms are validated for reuse. Lexical-only and minimal-plus-lexical require ten new runs after explicit approval. |
| 3 — Training-token curves | Deferred | Do not start until Phase 2.6 identifies a representation with a credible worst-group signal. |
| 4 — Taxonomy/representation scaling | Queued | Use the representation selected after the Phase 2.6 decision gate. |
| 5 — Matched architecture benchmark | Queued | Run only after a non-saturated parser configuration is identified. |

Last verification before this checkpoint: the corrected Phase-2 run completed
all 60 cells and Phase 2.5 completed ten frozen probes plus 15 new training
runs. The full unit suite reports 140 passing tests. Phase 2.6 preparation
validated all ten reused artifacts and all four manifests with zero exact
overlap; only registration/manifests exist in its output directory. The sealed
Stage-C suite has not been created or opened. The earlier
`runs/breadth_reinforcement_v1` artifact is retained for diagnosis only because
its fixed 40-step warmup confounds the cells. The working-tree research changes
have not yet been committed.

## Objective

Estimate the marginal robust-generalization gain—the Swedish *tangentens
lutning*—from additional language-form diversity, training tokens, model
capacity, and semantic taxonomy complexity. The aim is to learn where each
architecture is still improving, and only then run expensive full-agent
comparisons.

For a resource quantity \(x\) and robust accuracy \(A\), the local estimate
between adjacent doubling points is:

\[
s(x) = \frac{A(2x) - A(x)}{\log_2(2x/x)} = A(2x) - A(x).
\]

We report this as percentage points per doubling, with seed-level bootstrap
confidence intervals. The primary score is worst-group robust accuracy, not a
single average that can conceal brittle surface-form behavior.

## Non-negotiable experimental controls

1. **No evaluation text in tokenizer training.** The tokenizer corpus contains
   training inputs and targets only. Held-out wording, lexical items, and
   benchmark prompts stay entirely outside training and tokenizer fitting.
2. **Fixed external test set across conditions.** Changing the number of
   training templates must not change the difficulty or composition of the
   evaluation set.
3. **Matched budgets.** Report padded and non-padding tokens, updates,
   parameters, wall clock, and inference latency. Breadth comparisons hold
   total trainable tokens constant.
4. **Five independent seeds per point.** Report mean, standard deviation, and
   bootstrap 95% interval. Do not interpret a one- or two-seed difference as a
   slope.
5. **Frozen development/test split.** Tune only against a development set;
   open the final pressure test once per registered configuration.
6. **Layer-specific scores.** Keep semantic routing, protocol validity,
   execution-ready action, and grounded end-to-end episode success separate.

## Active sequence

### Phase 0 — Rebuild the clean semantic benchmark — Complete

Create `form_variation_v2` with a tokenizer trained solely on each condition’s
training partition. Build a fixed, untouched pressure suite with independent
groups for held-out templates, lexical shifts, syntax/order reversals,
discourse distractors, minimal contrasts, and unseen operands.

**Result:** `form_variation_v2` has a fixed 126-case external pressure suite.
Every K-specific manifest reports 0 exact train/tokenizer/test utterance
overlap and `PASS`; the unit suite checks this guarantee.

### Phase 1 — Form-diversity curve — Complete

Train the parser with 1, 2, 4, and 8 forms per intent, five seeds each and the
same update-token budget. Evaluate the fixed Phase-0 suite.

**Result:** the five-seed clean curve found no worst-group lift above chance.
Macro robust accuracy improves through K=4 but regresses at K=8. Full results,
paired slopes, and uncertainty are in [the Phase 0–1 report](form_variation_clean_v2.md).

**Decision rule:** if the upper 95% confidence bound is below 1 point per
doubling for two consecutive intervals, treat this scale axis as saturated for
the present model/data family.

### Phase 2 — Breadth versus reinforcement — Complete

At each of selected diversity levels, hold total tokens fixed while varying
examples per template. Compare few forms repeated often against many forms
seen less often.

**Question:** is improved generalization caused by semantic coverage itself,
or merely by more distinct tokens/examples?

**Result:** the corrected 60-cell factorial found a positive macro-robustness
transition from K=2 to K=4, no reliable reinforcement slope, and no stable
worst-group improvement. See the [Phase 2 report](breadth_reinforcement_v2.md).

### Phase 2.5 — Representation diagnostic and ablation — Complete; gate not passed

Determine whether the remaining failure is caused by an absent semantic
representation, a weak output/readout protocol, or insufficient contrastive
supervision. This phase is deliberately staged so a cheap diagnostic gates new
training compute.

#### Stage A — Frozen-checkpoint diagnostic

Reuse the five K=4/R89 and five K=8/R89 corrected-v2 checkpoints. At the hidden
state immediately before the parser response, fit frozen linear probes using
training forms only and evaluate them on the existing pressure suite:

1. classify the canonical operation;
2. recover the semantic role of each numeric span, with special attention to
   subtraction and reversed surface order;
3. measure minimal-contrast consistency alongside per-track accuracy.

The existing pressure suite is now a development suite for this phase; it must
not be presented as a newly sealed confirmation result.

**Diagnostic routing:**

- If intent is recoverable but normal decoding fails, prioritize a structured
  readout/typed-frame target rather than more language data.
- If intent is not recoverable, prioritize balanced minimal-contrast
  supervision.
- If intent is recoverable but argument roles are not, prioritize explicit
  semantic-role binding in the typed target.

#### Stage B — K=4 representation screen

At K=4 and the middle exposure level R89, compare this registered 2 × 2 design:

| Parser target | Ordinary training | Contrast-balanced training |
| --- | --- | --- |
| Operation only: `OP=ADD` | Existing corrected-v2 baseline | Minimal-contrast arm |
| Typed frame: `OP=SUBTRACT;A=345;B=456` | Typed-frame arm | Typed-frame + minimal-contrast arm |

`A` and `B` denote canonical semantic roles, not mention order. For example,
“Subtract 456 from 345” maps to `OP=SUBTRACT;A=345;B=456`. Contrast balancing
must replace examples within the matched budget rather than add training
tokens. Hold the tokenizer, model, optimizer, operand pairs, update-token
budget, warmup fraction, and five paired seeds fixed.

If the corrected-v2 baseline hashes and schedule remain unchanged, reuse its
five runs. The full screen therefore requires only 15 new parser runs. Score
operation selection, argument binding, full-frame exact match, each robustness
track, macro robustness, and worst-group robustness separately.

#### Stage C — Sealed confirmation

Select at most one Stage-B arm using the development suite. Compare that arm
against the baseline at K=4 and K=8, with five paired seeds, on a new frozen
suite containing unseen paraphrases, lexical shifts, syntax/order reversals,
discourse distractors, and minimal contrasts. Do not evaluate this suite until
the winning configuration and analysis have been registered. This stage is 20
new runs and is launched only if Stage B passes its gate.

**Continuation gate:** advance when the selected arm improves worst-group
robust accuracy by at least 10 percentage points over the paired baseline, its
paired bootstrap 95% interval excludes zero, and no individual robustness
track regresses by more than 5 points. If no arm passes, stop repetition/token
scaling and reassess the representation or model capacity before Phase 3.

**Result:** the Red-Green cycle is complete for canonical role binding, strict
typed-frame serialization, budget-preserving contrast balancing, frozen
response-boundary probes, operation/binding/exact-match scoring, training
fingerprints, the continuation gate, and three-way split validation. Stage A
and all 15 Stage-B runs completed. The detailed result is in the [Phase 2.5
report](phase25_representation_v1.md).

The CLI requires an explicit stage so the configuration cannot start training
by accident:

```powershell
# Safe: rewrite registration and manifests only.
.\.venv\Scripts\python.exe -m eval.form_variation `
  --config configs/phase25_representation.yaml --phase25-stage prepare

# Completed: frozen checkpoint probes; no base-model training.
.\.venv\Scripts\python.exe -m eval.form_variation `
  --config configs/phase25_representation.yaml --phase25-stage probe

# Completed: 15 new parser runs; do not repeat without a new registration.
.\.venv\Scripts\python.exe -m eval.form_variation `
  --config configs/phase25_representation.yaml --phase25-stage screen
```

### Phase 2.6 — Targeted minimal × lexical contrast coverage — Complete; rejected at sealed confirmation

Phase 2.5's minimal-contrast arm increased macro robustness in every seed, but
improved the worst-group minimum in only two. In the other seeds the limiting
track moved primarily to lexical shift. Phase 2.6 tests whether this is a
specific remaining coverage hole rather than a general representation ceiling.

At K=4/R89, use the same fixed tokenizer, 144-example budget, model, optimizer,
800 updates, 80 warmup updates, operand pairs, pressure suite, and five paired
seeds:

| Minimal-contrast factor | Lexical-contrast factor | Arm | Execution state |
| --- | --- | --- | --- |
| No | No | Baseline | Reuse five validated Phase-2 runs |
| Yes | No | Minimal contrast | Reuse five validated Phase-2.5 runs |
| No | Yes | Lexical contrast | Five new runs |
| Yes | Yes | Minimal + lexical contrast | Five new runs |

Each active factor replaces one standard example per operand/operation cell;
it never adds examples or updates. Training lexical templates are distinct
from the development pressure text. Report paired minimal and lexical main
effects both with and without the other factor, plus the factorial interaction,
for worst-group and macro robust accuracy.

Only the two new arms are eligible for advancement. Apply the existing gate:
at least +10 pp worst-group gain over baseline, paired bootstrap lower bound
strictly above zero, and no robustness-track regression greater than 5 pp.
Do not open the sealed suite unless a new arm passes.

The runner is guarded by an explicit stage:

```powershell
# Completed and safe: registration/manifests/source validation only.
.\.venv\Scripts\python.exe -m eval.form_variation `
  --config configs/phase26_contrast_coverage.yaml --phase26-stage prepare

# Completed: trained the two new arms, ten runs total.
.\.venv\Scripts\python.exe -m eval.form_variation `
  --config configs/phase26_contrast_coverage.yaml --phase26-stage screen
```

The screen selected the minimal-plus-lexical arm (+21.1 pp worst-group delta,
CI [+3.3, +38.9]), but the independent fresh-seed sealed confirmation rejected
it. At K=4 its delta was only +3.3 pp (CI [-6.7, +16.7]) and syntax order
regressed 14.5 pp. At K=8 its worst-group delta was -26.7 pp (CI [-33.3,
-17.8]), driven by a -37.8 pp discourse regression. See the [Phase 2.6
report](phase26_contrast_coverage_v1.md).

### Phase 2.7 — Failure audit — Next, no training

Before another curriculum change, produce a case-level audit of the sealed
K=4/K=8 predictions: operation confusion matrices, exact examples responsible
for each worst-track regression, and a per-template/operation decomposition.
This is analysis-only and must not use the sealed suite to tune a new training
intervention. Register any follow-up curriculum against a new sealed suite.

### Phase 3 — Training-token curves — Deferred pending a confirmed Phase 2 intervention

For low diversity (K=1) and high diversity (K=8), train at 0.1x, 0.3x, 1x,
3x, and 10x the Phase-1 token budget. Use the same test suite.

**Question:** does diversity improve sample efficiency, the eventual ceiling,
or both?

### Phase 4 — Taxonomy and representation scaling

Increase the canonical semantic inventory in controlled stages: arithmetic
operations, comparisons/equality, argument order, then compositional/nested
frames and dialogue context. Evaluate operation selection and argument binding
separately.

**Question:** where does a compact semantic representation stop compressing
the interpretation space effectively?

### Phase 5 — Matched architecture benchmark

After Phase 0–4 identify a non-saturated parser configuration, compare:

1. semantic parser → typed IR → deterministic controller/executor;
2. SmolLM2 ReAct;
3. a newly trained, corpus-validated scratch ReAct model.

Match task distribution and trainable-token budget as closely as practical.
Use the architecture benchmark’s separate routing, protocol, executable-action,
and grounded-episode tracks. The historic 300M scratch checkpoint remains a
diagnostic reference only and cannot be used for a scientific head-to-head
claim.

## Findings retained as priors, not conclusions

| Finding | Consequence for this roadmap |
| --- | --- |
| The first parser pilot improved held-out-form accuracy from 35.2% at K=1 to 54.9% at K=8, with perfect familiar-form and unseen-operand accuracy. | It motivated the clean rerun. The clean five-seed curve confirmed familiar-form mastery but not worst-group form invariance; see the Phase 0–1 report. |
| The parser is perfect on observed forms but weak under held-out wording and discourse distractors in the 36-case pressure test. | Use worst-group robust accuracy and independent pressure groups as primary outcomes. |
| SmolLM2-135M ReAct reached 66.7% shared routing but only 54.2% execution-ready accuracy; it failed discourse distractors. | Do not collapse intended operation and usable executable action into one score. |
| SmolLM2 showed 0% prior contamination in the earlier benchmark but weak retrieval, abstention, and multi-document behavior. | Retain it as a practical baseline; test it on matched tracks instead of treating direct computation as evidence of full agent competence. |
| The historical 300M scratch run reached 47.8% on an outdated evaluation configuration, but its later corpus failed contamination validation. | Preserve artifacts and exclude the result from comparative claims until a clean rerun exists. |

## Deliverables per completed phase

- Versioned YAML configuration and split manifest.
- Per-seed metrics, timings, token counts, checkpoint paths, and case-level
  predictions.
- A short result note that gives slope estimates and confidence intervals,
  including negative or flat results.
- An update to `docs/architecture_benchmark.md` when a new architecture or
  benchmark track is added.

## Immediate next action

Phase 2.6 is complete and rejected by sealed confirmation. Execute the Phase
2.7 read-only failure audit next; do not launch Phase 3 or another curriculum
training run until it has produced a new, independently sealed registration.
