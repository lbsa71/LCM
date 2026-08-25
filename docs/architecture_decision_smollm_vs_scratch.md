# LCM Architecture Decision: SmolLM2 vs From-Scratch

**Date:** 2026-08-25
**Context:** After 9 experiments on SmolLM2-135M (Grounded Accuracy 45%, Prior Contamination 0%), evaluating whether to continue with pretrained fine-tuning or resume from-scratch pretraining.

---

## SmolLM2 (Fine-tuned Pretrained Base)

### Pros
1. **Fast iteration** — 26 min per full experiment cycle (data gen → SFT → eval). We ran 9 experiments in a single day.
2. **Direct computation works perfectly** — Suite A/B/H all at 100%. The model learned `<ACTION> MATH` routing flawlessly, proving the RDL protocol *can* override pretraining.
3. **Zero prior contamination so far** — 0.0% PCR across all experiments. It never answered "Paris" when the document said "Lyon". Empirically, SFT *is* suppressing priors.
4. **Pre-learned subword structure** — The tokenizer already handles arbitrary user input (names, numbers, punctuation) without us building one from scratch.
5. **Clear scaling path** — 360M config is ready; 1.7B SmolLM2 exists. We can test whether retrieval routing improves with capacity without re-engineering infrastructure.

### Cons
1. **The retrieval bottleneck may be intrinsic** — 40% on single-hop, 0% on multi-doc. The model's pretrained tendency is to *generate text*, not emit `SEARCH "entity" LIMIT 3`. We're fighting gradient inertia from 600B+ pretraining tokens.
2. **ABSTAIN is broken (20/55 failures)** — Pretrained LLMs are optimized to *always produce a helpful response*. Teaching one to say "I don't know" contradicts its deepest training signal.
3. **Latent priors are unauditable** — 0% PCR on our 12-item counterfactual bank doesn't prove zero contamination. There are billions of facts in those weights we can't test.
4. **Philosophically contradicts the LCM thesis** — We're trying to prove that a model *doesn't need world knowledge*, using a model *stuffed with world knowledge*. Any positive result has the asterisk: "but maybe SmolLM2 already knew the answer."
5. **Loss floor is suspicious** — Loss converged to ~0.01–0.07 but retrieval suites are at 0–40%. The model may be memorizing surface patterns of the training trajectories rather than learning the procedural routing algorithm.

---

## From-Scratch (SyntheticTransformer, zero pretraining)

### Pros
1. **Epistemic purity by construction** — The model has literally *zero* world knowledge. No asterisks, no "maybe it already knew." If it answers correctly, it's because it followed the procedure.
2. **No adversarial gradient inertia** — Every weight learns *only* RDL routing. No 600B-token prior pulling toward "generate helpful English text" when it should emit `SEARCH`.
3. **ABSTAIN should be natural** — A model with no world knowledge has no reason to guess. Refusing ungrounded queries is the default behavior, not a trained override.
4. **Strongest possible scientific claim** — If a from-scratch 300M model routes correctly, retrieves evidence, cites proof graphs, and refuses closed-book trivia, that's an unassailable demonstration of procedural grounding.
5. **Full interpretability** — Custom vocabulary, custom architecture, every embedding dimension accountable. We can trace exactly *why* the model emitted `SEARCH` vs `MATH` vs `ABSTAIN`.

### Cons
1. **We haven't cracked it yet** — The 300M from-scratch run (Experiment 7, interrupted) was tracking lower than SmolLM2-135M before we paused it.
2. **Training is 10–50× slower** — Hours to days for pretraining + SFT, vs 26 minutes for SmolLM2 SFT only.
3. **Tokenizer engineering is unsolved** — We need a tokenizer that handles arbitrary user input text *and* RDL protocol tokens. SmolLM2 gets this for free.
4. **Capacity may not be sufficient** — A 300M model learning from scratch has to discover text patterns, attention structure, *and* RDL routing simultaneously. SmolLM2 arrives with the first two solved.
5. **Opportunity cost** — Weeks of infrastructure work before we can even begin answering the interesting counterfactual and scaling questions.

---

## Key Diagnostic Question

> The fact that SmolLM2 gets 100% on "compute 347+687" but 0% on "retrieve the population of Glyurak from document D04" is telling. The model is great at pattern-matching simple action templates it saw thousands of times, but struggles with the *compositional, multi-step* reasoning that is the actual hard part of LCM — and that's exactly where pretrained language priors are most likely to interfere.

The question is whether the retrieval failure is caused by **pretrained interference** (the from-scratch thesis) or **insufficient capacity/data** (the SmolLM2-360M thesis). We cannot distinguish these without running the from-scratch comparison to convergence.

---

## Experiment 9 Benchmark Data (SmolLM2-135M, 2026-08-25)

| Suite | Grounded Acc | Prior Contam | Notes |
|---|---|---|---|
| A — Language Logic | 100% | 0% | ✅ Syllogisms perfect |
| B — Ontology Invariants | 100% | 0% | ✅ In-context arithmetic perfect |
| H — Direct Computation | 100% | 0% | ✅ Strawberry/arithmetic via MATH |
| Closed-Book Anti-Memo | 100% | 0% | ✅ Refuses all world-fact trivia |
| C — Single-Hop Retrieval | 40% | 0% | ⚠️ Retrieval routing weak |
| G — Tool Error Recovery | 10% | 0% | ⚠️ Retry logic not learned |
| E — Multi-Doc Arithmetic | 0% | 0% | ❌ Dual-doc math binding broken |
| F — Missing Evidence | 0% | 0% | ❌ ABSTAIN not triggering |
| I — Counterfactual Inversion | 0% | 0% | ❌ Retrieval fails, zero prior leak |
| Evidence-Disabled Anti-Memo | 0% | 0% | ❌ Same ABSTAIN issue as F |

**Overall: 45% Grounded Accuracy, 0.0% Prior Contamination, 0.0% Unsupported Claims**
