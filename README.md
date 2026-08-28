# LCM: Fact-Externalized Language Agents

> **Research status: paused after feasibility study — 2026-08-28.**
>
> This repository is preserved as a research record and stepping stone. It is
> not a production model, a validated general-purpose agent, or evidence that a
> fact-free foundation model is feasible. No training is currently authorized.
> Resumption requires the externally anchored kill test summarized below.

## Abstract

LCM investigated whether contingent world knowledge could remain outside model
weights while a small neural policy learned language interpretation,
procedural reasoning, planning, and tool use. The prototype combines synthetic
counterfactual worlds, deterministic `SEARCH` and `READ` tools, constrained
`MATH` and `FILTER` operations, structured agent messages, proof-addressed
evaluation, and two model
families: a compact semantic parser and a direct ReAct-style policy.

The study found one narrow positive result. Already-pretrained SmolLM2-135M and
SmolLM2-360M models followed changed document values on 8/10 and 9/10 paired
counterfactual probes, respectively, without reusing the previous answers.
The broader results were negative or inconclusive. A scratch-trained compact
parser mastered familiar forms but remained near chance on its weakest held-out
language groups; additional form breadth, repetition, counterbalancing,
consistency supervision, and modest width did not produce sealed semantic
invariance. Increasing the pretrained policy from 135M to 360M parameters
improved a repaired development probe by only 1.82 percentage points, with an
interval spanning zero, at 19.39 times the reported training-loop time. Both
models scored zero on the repaired multi-hop and retrieval-abstention tracks.

A later validity audit found incomplete demonstrations, missing evidence,
absent suite coverage, an unenforced seed, and permissive grounding paths in
the historical pretrained-agent pipeline. Repaired frozen-model probes improve
the measurement but cannot retroactively validate the training corpus. The
central claim—a broadly capable model trained from random weights on synthetic,
fact-externalized language—therefore remains untested. The present evidence
supports further study of evidence-auditable runtimes, not continued scaling of
this model family.

## Research questions

The project combined three hypotheses whose evidence must remain separate.

| Hypothesis | Current conclusion |
| --- | --- |
| A neural policy can use external evidence inside a deterministic runtime. | Supported only on a small value-swap diagnostic using already-pretrained models. |
| A compact semantic bottleneck can map varied language to typed executable actions. | The tested 96-wide scratch parser is rejected as a robust general parser. Larger or pretrained task-specific parsers remain open. |
| Synthetic, fact-externalized training from random weights can yield a broadly capable small agent. | Not tested at a defensible data scale or with a clean shared comparator. |

The pretrained experiments test evidence-conditioned behavior, not fact-free
learning: SmolLM2 already contains broad linguistic and factual pretraining.

## System and method

LCM separates four responsibilities:

1. a neural policy interprets language and proposes plans or actions;
2. a deterministic shell manages protocol state, tools, permissions, and
   budgets;
3. an external environment holds documents and contingent world state; and
4. an evaluator checks evidence against hidden proof graphs and expected final
   state.

The main experimental branches were:

- scratch semantic parsing of `ADD`, `SUBTRACT`, and `COMPARE` under controlled
  language-form variation;
- direct pretrained ReAct policies based on SmolLM2-135M and SmolLM2-360M;
- a historical synthetic-only scratch ReAct model, retained as diagnostic
  evidence only; and
- paired counterfactual value swaps intended to distinguish evidence use from
  fixed-answer reproduction.

Most parser studies used five paired seeds, matched update budgets, held-out
pressure groups, paired bootstrap intervals, continuation gates, and a one-time
sealed confirmation. Software tests enforce implementation invariants; passing
them does not by itself establish corpus or experimental validity.

## Positioning

The project extends the controlled-skill tradition represented by
[bAbI](https://arxiv.org/abs/1502.05698), but treats such tasks as diagnostics,
not substitutes for broad language. Its typed-representation branch is closer
to executable [dialogue dataflow](https://www.microsoft.com/en-us/research/blog/dialogue-as-dataflow-a-new-approach-to-conversational-ai/)
than to a closed universal intent taxonomy. Contemporary pretrained specialists
such as [FunctionGemma-270M](https://deepmind.google/models/gemma/functiongemma/)
make an external small-model baseline mandatory. Recent synthetic-environment
work likewise emphasizes deep, stateful, verifier-backed worlds rather than
surface-form volume. LCM here is unrelated to Meta's separate
[Large Concept Model](https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/)
project.

## Results

| Experiment | Result | Interpretation |
| --- | --- | --- |
| Form breadth, K=1–8 | Familiar form/new operands: 100%. Worst robust group: 23.3–33.3%. Macro peak: 46.3% at K=4. | The parser learned familiar surface families, not a stable operation-invariant boundary. |
| Repetition tail | At K=8, macro slope −0.25 pp/doubling, 95% CI [−5.04, +4.54]; worst-group slope −1.26, CI [−3.79, 0.00]. | No sustained positive marginal gain; the repetition axis was closed. |
| Parser sealed confirmation | Worst-group change −2.78 pp, CI [−15.00, +10.00]; held-out templates −32.78 pp. | Consistency increased agreement but not correctness: more consistent errors. |
| Shared architecture probe | Parser average routing 60.4%; SmolLM2-135M routing 66.7%, execution-ready 54.2%. | Both remained brittle under discourse variation; routing overstated usable action accuracy. |
| Repaired pretrained probe | 135M: 48.18%; 360M: 50.00%; delta +1.82 pp, world-cluster interval [−2.20, +5.00]. | No reliable capacity gain. |
| Training-loop cost | 26.3 minutes versus 8.5 hours. | The 360M run cost 19.39× more reported time without a reliable overall benefit. |
| Paired value swaps | 135M: 17/20 and 8/10 complete pairs; 360M: 18/20 and 9/10 pairs; zero old-answer reuse. | Credible but narrow evidence-conditioned behavior in pretrained models. |
| Repaired core tracks | Both models: 0% multi-hop and 0% retrieval-abstention. | General-purpose readiness was not demonstrated. |

The historical 300M scratch checkpoint reached 47.8% on an obsolete aggregate
evaluation, but its corpus failed later validity controls and its interface is
not comparable with the current benchmark. It cannot support a
scratch-versus-pretrained ranking.

## Validity limits

The Phase 5 audit found that the historical 184,000-trajectory pretrained
corpus contained:

- 16,000 missing-evidence and 8,000 evidence-disabled demonstrations ending at
  `SEARCH`, without an observation or terminal answer;
- sampled counterfactual tasks whose required documents were absent;
- no multi-hop training or evaluation cases;
- evidence-disabled evaluations that still retained documents;
- substantial familiar-question reuse; and
- a configured 135M seed that was not enforced.

The audit repaired several evaluation and corpus-validation paths. Before any
new training, known blockers still include teacher-side search-hit injection,
fabricated recovery observations, fallback operands or answers, possible loss
of terminal targets through truncation, and runtime enforcement that cited
evidence was actually observed and satisfies the proof graph.

Consequently:

- historical failures do not establish a universal architectural limit;
- repaired development probes are not sealed confirmations;
- cases sampled from a few synthetic worlds are not independent evidence;
- protocol-valid output is not equivalent to a true or supported answer; and
- the results provide no defensible timeline to a general-purpose agent.

## Conclusion

The broad LCM thesis is neither confirmed nor cleanly falsified. It has not
received the language diversity, token scale, valid supervision, or external
baselines required for a fair test. Continuing unchanged would require
substantially more data engineering and compute while competing with strong
pretrained small models.

Two narrower conclusions are justified:

1. the tested compact scratch parser did not acquire robust semantic invariance
   through additional templates, repetition, modest width, or the tested
   consistency objectives; and
2. small pretrained models can follow changed external evidence in a narrow
   controlled setting, but this did not yield reliable abstention, multi-hop
   reasoning, or general language robustness.

The project is therefore paused. Its continuing value is as a reproducible
record of negative results, synthetic-data validity failures, evidence-aware
evaluation machinery, and a possible foundation for a model-independent
verified-agent benchmark.

## Conditions for resumption

The proposed next step is a capped, externally anchored kill test—not another
open-ended scaling run. It first repairs every oracle, fabricated-observation,
fallback-answer, silent-truncation, and runtime-proof path. It then compares a
contemporary small function-calling model, SmolLM2-135M direct ReAct, the same
backbone with typed executable dataflow, and deterministic/strong-model
ceilings on identical sealed tasks. A scratch arm is earned only if the
system-level comparison first shows a clear advantage.

The decision-grade test is capped at 14 working days, 120 engineering hours,
and approximately 50 local GPU-hours before the separately approved scratch
extension. Candidate continuation gates include a +10 point worst-group gain
or a 3× efficiency advantage at matched reliability, no core-group regression
above five points, at least 90% correct absent/withheld-evidence behavior, and
at least 70% multi-hop final-state success.

The complete protocol and stopping rules are in
[docs/kill_test.md](docs/kill_test.md). No part of that experiment has been
registered or launched.

## Research record and reproduction

The curated documentation index is [docs/README.md](docs/README.md). Principal
evidence includes:

- [clean form-variation curve](docs/history/form_variation_clean_v2.md);
- [breadth and reinforcement study](docs/history/breadth_reinforcement_v2.md);
- [completed training-token curve](docs/history/phase3_scaling_complete.md);
- [parser sealed confirmation](docs/history/phase4_stagec_invariance.md);
- [shared architecture benchmark](docs/history/architecture_benchmark.md);
- [Phase 5 validity audit](docs/history/phase5_validity_audit.md); and
- [feasibility verdict](docs/history/phase5_feasibility_verdict.md).

Create a local environment and run the software verification suite:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest tests/unit/
```

Historical configurations remain under `configs/` for reproducibility. Large
checkpoints, ledgers, and generated corpora remain local under ignored `runs/`
directories; they are evidence for the documented experiments, not endorsed
production models or authorized future runs. See
[docs/artifacts.md](docs/artifacts.md) for the retention policy and provenance
map.
