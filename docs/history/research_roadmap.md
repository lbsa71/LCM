# Archived research roadmap: robust-generalization slopes

**Date:** 2026-08-27

**Status:** Historical sequence, superseded by the project pause on 2026-08-28.
The current synthesis is in the repository [README](../../README.md); any future
work requires the deferred [kill-test protocol](../kill_test.md). References to
an active or next phase below describe the state at the time of writing.

**Historical status:** Phase 5 capacity training complete; benchmark-validity repair takes priority
**Supersedes:** the implicit “continue a large from-scratch run” direction in
the prior architecture decision. It does not supersede the evidence recorded
there; it changes the order in which we spend training compute.

## Resume checkpoint — 2026-08-27

The 360M run finished in 8.50 hours. The frozen legacy-runtime and repaired-data
replays are complete. A new corpus audit found that all old sampled counterfactual evidence
is absent, all 24,000 retrieval-abstention demonstrations are unfinished, and
the multi-hop suite is empty. **Do not treat the previous readiness failures
as architectural limits or launch another capacity run.** The first repairs
have regression tests. See [the validity audit](phase5_validity_audit.md) and
[the capacity experiment](phase5_pretrained_capacity.md). Historical artifacts
are retained unchanged apart from explicitly documented provenance corrections.

The repaired 110-case probe gives 135M 48.18% and 360M 50.00% strict grounded
success; the +1.82 pp difference has a world-cluster interval spanning zero.
Both reach 90% on counterfactual evidence when its documents actually exist,
but both still fail multi-hop and retrieval-abstention. The paired value-swap
probe passed 8/10 complete pairs for 135M and 9/10 for 360M, supporting real
evidence dependence on these cases. See the [feasibility checkpoint](phase5_feasibility_verdict.md).
Latest verification:
217 unit/integration tests passed. No further training has been launched.

### Earlier completed sequence

Phase 2 is complete through the fixed-template capacity interaction, and that
curriculum branch is retired. Phase 3 completed the full sequential curve
through R889. The local K=8 R178-to-R356 macro gain did not persist:
R356-to-R889 macro and worst-group intervals cross or touch zero at both K=1
and K=8. Repetition scaling is closed as flat/inconclusive for semantic
invariance. Phase 4 Stage A is complete. Its multitask arm produced a strong
worst-group gain but missed the no-regression gate on discourse distractors;
a read-only failure audit is complete. Stage B will use a fresh development
suite to test binding-loss weight and scaffold counterbalancing.

| Phase | State | Resume note |
| --- | --- | --- |
| 0 — Clean semantic benchmark | Complete | Fixed 126-case pressure suite; all split manifests pass with zero exact overlap. |
| 1 — Form-diversity curve | Complete | Twenty five-seed runs; familiar forms mastered, worst-group robustness remains near chance. |
| 2 — Breadth versus reinforcement | Complete | Corrected 60-cell v2 factorial is the scientific result; repetition has no reliable slope. |
| 2.5 — Representation diagnostic and ablation | **Complete; rejected** | Minimal contrasts improved the screen mean but not its confidence-bound gate; typed targets regressed distractor tracks. |
| 2.6–2.10 — Coverage, audit, augmentation, capacity | **Complete; rejected** | Sealed confirmation exposed deterministic cue policies; replacement, augmentation, extra exposure, and modest width did not repair worst-group robustness. |
| 3 — Training-token curves | **Complete; repetition closed** | K=8 R356→R889 macro -0.25 pp/doubling, CI [-5.04, +4.54]; worst -1.26, CI [-3.79, 0.00]. |
| 4 — Taxonomy/representation scaling | **Complete; rejected** | Consistency improved agreement but failed sealed correctness and robustness. |
| 5 — Architecture feasibility | **Validity repair; frozen weights retained** | 360M training finished. Old scores are diagnostic-only: repair absent evidence, incomplete demonstrations, and missing suite coverage before ranking architectures. |

The earlier Phase-3 checkpoint had 166 passing tests; the first Phase-5 validity
repair pass has 198. The corrected Phase-2 run completed all 60 cells; Phases
2.5–2.10 and the complete Phase-3 curve through R889 are finished. The earlier
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

**Complete:** the audit found that the K=8 combined arm systematically maps
the sealed discourse `COMPARE` cases to `SUBTRACT` (30/30) and `SUBTRACT`
cases to `ADD` (30/30). This is a shared-scaffold shortcut, not a residual
lexical gap. The next intervention must vary distractor clauses explicitly,
with a new sealed suite that is never used for development.

### Phase 2.8 — Counterfactual discourse coverage — Complete; rejected

At K=8/R89, compare fresh five-seed baselines against a fixed-budget curriculum
that replaces 25% of standard examples per operand/operation cell with
training-only distractor clauses. Clause polarity and requested operation are
counterbalanced, so no cue can predict `ADD`, `SUBTRACT`, or `COMPARE` by
itself. The screen retains the existing development suite; advancement requires
the existing +10 pp/positive-CI/no-regression gate. A passing screen must be
confirmed on a newly created sealed discourse suite, never the audited suite.

The screen improved discourse distractors by +20.0 pp (CI [+7.8, +32.2]) but
regressed held-out templates by 17.2 pp and syntax order by 20.0 pp; worst-group
accuracy fell 7.2 pp. It failed the gate, so no sealed suite was created. See
the [Phase 2.8 report](phase28_counterfactual_discourse_v1.md).

### Phase 2.9 — Replacement versus augmentation — Complete; rejected

Reuse Phase 2.8's validated baseline and replacement cells. Add two five-seed
cells that retain all standard K=8 forms and append the counterbalanced
discourse examples: one at the original 1,600 updates and one with updates
scaled by 1.25x to preserve per-example exposure. This distinguishes deletion
cost, diversity dilution, and additional-compute effects without reopening a
sealed suite. Advance only if an augmentation arm passes the existing gate.

Neither augmentation arm passed. Fixed-update augmentation changed worst-group
accuracy by -5.6 pp; matched-exposure augmentation changed it by -17.8 pp.
The extra 25% updates had a paired -12.2 pp effect (CI [-18.9, -4.5]), showing
that additional optimization strengthens the specialized shortcut. See the
[Phase 2.9 report](phase29_discourse_augmentation_v1.md).

### Phase 2.10 — Capacity interaction — Complete; rejected

Reuse Phase 2.9's 96-wide baseline and fixed-update augmentation cells. Train
fresh paired baseline and augmentation cells at hidden size 144 with all other
controls fixed. Estimate the width x augmentation interaction for worst-group
and macro robustness. The wide augmentation cell may advance only relative to
the paired wide baseline under the existing gate; a passing screen requires a
new sealed confirmation.

The width x augmentation interaction was -4.5 pp for worst-group accuracy (CI
[-32.8, +32.2]) and +1.9 pp macro (CI [-7.7, +14.7]). Wide augmentation still
regressed held-out forms and failed the gate. See the [Phase 2.10
report](phase210_capacity_interaction_v1.md). This closes the fixed-template
curriculum branch.

### Phase 3 — Training-token curves — Complete; repetition closed

For low diversity (K=1) and high diversity (K=8), train at 0.1x, 0.3x, 1x,
3x, and 10x the Phase-1 token budget. Use the same test suite.

**Question:** does diversity improve sample efficiency, the eventual ceiling,
or both?

Use a sequential extension rather than immediately paying for the full 10x
tail. Reuse K=1/K=8 R178 results and train R356 at five paired seeds. Continue
to R889 only if at least one breadth has a macro per-doubling slope whose
paired-bootstrap lower bound is positive, with no worst-group mean regression
greater than 5 pp. Otherwise close repetition scaling as flat or harmful.

**Stage A result:** K=8 passed the gate: macro robustness rose +4.00 pp per
doubling (paired 95% CI [+1.33, +6.89]) and worst-group accuracy rose +0.56 pp.
K=1 was inconclusive. Run the registered R889 cells for both breadths, then
compare R356→R889 slopes and the full curve. See the [Stage A report](phase3_scaling_stage_a.md).

**Terminal interpretation registered before R889 completion:** a positive
macro lower bound with no >5 pp worst-group mean regression establishes only
average sample-efficiency improvement. Repetition counts as repairing the
semantic-invariance failure only if the final worst-group slope also has a
strictly positive lower bound. A confidence interval containing zero is
flat/inconclusive and a negative upper bound is harmful. R889 closes the
repetition axis in every case; Phase 4 follows rather than extending the tail.

**Final result:** neither K=1 nor K=8 has a positive terminal macro or
worst-group lower bound. K=8 macro moved -0.25 pp per doubling (CI [-5.04,
+4.54]) and worst group -1.26 pp (CI [-3.79, 0.00]) from R356 to R889. The
Stage-A improvement was a local bump rather than a sustained tangent. See the
[complete Phase 3 report](phase3_scaling_complete.md).

### Phase 4 — Taxonomy and representation scaling

Increase the canonical semantic inventory in controlled stages: arithmetic
operations, comparisons/equality, argument order, then compositional/nested
frames and dialogue context. Evaluate operation selection and argument binding
separately.

**Question:** where does a compact semantic representation stop compressing
the interpretation space effectively?

#### Stage A — Separate semantic supervision from frame serialization

Before expanding the taxonomy, test whether Phase 2.5's typed-frame failure
was caused by the representation or by autoregressive serialization. The
original corpus confounds B-first order with SUBTRACT, so replace forms within
each operation/operand cell to obtain a fixed-budget 50/50 mention-order
distribution. At K=8/R178 train a matched generative baseline, an end-to-end
operation classifier, and an operation classifier with an auxiliary canonical
A/B mention-order head. This adds 15 runs and keeps the data, updates,
tokenizer, architecture width, optimizer, pressure suite, and seeds fixed
across all three arms. The old Phase-2 cell is reference-only.

Advance at most one arm under the existing +10 pp worst-group, positive paired
CI, and no >5 pp track-regression gate. A passing arm requires a newly frozen
sealed confirmation before equality, nested frames, or dialogue context are
added. See the [Phase 4 Stage-A registration](phase4_bottleneck_screen_v1.md).

**Stage-A result:** operation-only gained +12.22 pp worst-group accuracy (CI
[+2.22, +22.22]) but regressed held-out templates 5.55 pp. Operation plus
binding gained +25.55 pp (CI [+12.22, +33.33]) and +9.44 pp macro (CI [+5.33,
+16.00]) but regressed discourse 6.67 pp. Neither arm passes; no sealed suite
was created. The multitask arm is retained as the strongest lead. Before any
new training, audit case-level discourse operation confusions and the repeated
minimal-contrast binding collapse. See the [Phase 4 Stage-A report](phase4_bottleneck_screen_v1.md).

#### Stage B — Binding-weight × scaffold-coverage factorial — Complete; no gate pass

The [read-only audit](phase4_failure_audit.md) found that the discourse
regression is entirely driven by one seed, while all five encoders and a
refitted frozen probe map every minimal-contrast case to the wrong binding
class. Register a fixed-budget 2 × 2 at K=8/R178:

| Binding-loss weight | Ordinary balanced corpus | Scaffold-counterbalanced corpus |
| ---: | --- | --- |
| 1.0 | Reuse Stage-A multitask checkpoints | Five new runs |
| 0.25 | Five new runs | Five new runs |

Every corpus must remain 50/50 A-first/B-first within each operation/operand
cell. Scaffold examples replace, never append. Evaluate the reused and new
arms on a newly frozen development suite with unseen text and balanced binding
labels; the Stage-A development suite is audit-only from this point. Compare
all arms to the matched generative checkpoints on the new suite using the same
+10 pp worst-group, positive paired-CI, and no >5 pp track-regression gate.
Select at most one arm. Only a passing arm permits creation of a new sealed
confirmation suite.

All 15 new runs completed. The weight-0.25 scaffold arm gained +12.67 pp macro
operation accuracy and +55.56 pp on minimal contrasts, but only +1.11 pp
worst-group accuracy (CI [-16.66, +15.55]) and regressed syntax by 10.00 pp.
No arm passed, and no sealed suite was created. A read-only frozen-probe audit
found no latent linear readout solution on lexical or syntax failures. See the
[Stage-B result](phase4_stageb_factorial.md) and [failure audit](phase4_stageb_failure_audit.md).

#### Stage C — Paired representation consistency — Complete; rejected

Reuse the weight-0.25 scaffold arm as the no-consistency lead and pair each
A-first training form with one B-first form from the same operation/operand
frame. Keep 16 sequences per update and 3,200 updates fixed. Train consistency
weights 0.25 and 1.0 (five paired seeds each), aligning same-intent operation
logits while leaving the binding head separately supervised. Re-evaluate the
matched generative baseline, reused lead, and both new arms on a third fresh,
binding-balanced development suite. Apply the unchanged worst-group gate. This
is ten new runs and directly tests the representation-invariance hypothesis.

All five consistency-0.25 seeds completed on 2026-08-26. The runner was then
stopped at the cell boundary; no consistency-1.0 seed completed. Restarting the
registered Stage-C screen validates and skips the first five runs before
training the remaining cell.

The remaining cell resumed on 2026-08-27 after all 177 unit tests and the
registered source, pairing, exposure, and development-suite controls passed.

All ten runs completed. Consistency-0.25 passed against the generative baseline
with a +26.67 pp worst-group gain (CI [+10.00, +41.11]) and no >5 pp track
regression. Consistency-1.0 failed on a 6.67 pp syntax regression. The direct
consistency-0.25 effect versus the no-consistency lead was only +1.11 pp (CI
[-7.78, +12.22]), so mechanism attribution is explicitly unresolved.

The sealed confirmation uses frozen baseline, lead, and selected checkpoints,
six new operand pairs, and both mention orders for every operation/frame: 216
fresh cases total. Apply the unchanged primary gate once; paired-form agreement
is secondary diagnostic evidence and cannot rescue a gate failure.

**Sealed result:** the candidate changed worst-group accuracy by -2.78 pp (CI
[-15.00, +10.00]) and regressed held-out templates by 32.78 pp. Paired-form
agreement improved by 9.56 pp (CI [+3.56, +14.22]), but paired correctness was
flat (+2.00 pp, CI [-4.44, +7.33]). This closes the compact 96-wide parser
branch: the intervention made errors more self-consistent without making the
semantic decisions reliably correct.

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

#### Stage A — Existing-checkpoint learning curves

Before paying for another training run, evaluate the saved SmolLM2-135M ReAct
checkpoints at SFT steps 1,000 and 2,000 on the same held-out suites already
used at step 3,000. Preserve suite-level grounded success, failure taxonomy,
latency, and approximate samples. The legacy scratch-300M curve remains useful
only to diagnose that run; its contaminated corpus forbids a family-level
comparison.

For a pragmatic general-purpose readiness signal, require both breadth and a
non-flat terminal slope: at least 70% overall grounded success, at least 40% on
each retrieval/computation, missing-evidence, recovery, and counterfactual core
suite, and at least +5 pp overall from step 2,000 to 3,000. Failure rejects the
135M checkpoint as usable or fast-converging, not the pretrained architecture
family. If it fails, the next discriminating run is the matched 360M pretrained
adapter; only after that capacity test decide whether a clean scratch rerun is
worth its substantially larger compute budget.

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

The frozen replays and paired value-swap diagnostic are complete. Prepare and
register the [small complete-trajectory/control experiment](phase5_next_control.md)
on the 135M baseline, including live tool-observation replay and tokenized
terminal-target coverage before training. Do not scale capacity on the invalid
corpus, reopen Stage-C confirmation, or return to fixed-template parser scaling.
The three-way architecture ranking, repaired multi-checkpoint learning curve,
and any general-purpose extrapolation remain unresolved.
