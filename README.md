# LCM — Synthetic-Only Agentic Language Model Proof-of-Concept

A small language model architecture trained exclusively from random weights on deliberately constructed synthetic language. It demonstrates that **procedural competence (planning, tool use, search, code synthesis, verification) can be learned separately from contingent factual memorization**.

---

## Key Principles

1. **Zero External Contamination**: Trained exclusively on procedurally generated synthetic language and nonce entities. No Wikipedia, Common Crawl, or pretrained checkpoints.
2. **Mutable-World Anti-Memorization**: Contingent facts and entities are randomized between worlds, making weight memorization unhelpful or penalized.
3. **Deterministic ReAct Shell**: The deterministic shell (not the neural weights) owns state, tool execution, permissions, sandboxing, evidence provenance, and limits.
4. **Epistemic Enforcement**: Final answers requiring retrieval that lack valid ground-truth evidence lines are scored as failures even if the text matches.

---

## Architecture

```
                    ┌───────────────────────┐
                    │ Synthetic World Spec  │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ World / Corpus        │
                    │ Generator             │
                    └───────────┬───────────┘
                                │
                ┌───────────────┴────────────────┐
                ▼                                ▼
      ┌──────────────────┐             ┌─────────────────┐
      │ Language Corpus  │             │ Agent Episodes  │
      └─────────┬────────┘             └────────┬────────┘
                │                               │
                ▼                               │
      ┌──────────────────┐                      │
      │ Base Pretraining │                      │
      └─────────┬────────┘                      │
                │                               │
                └──────────────┬────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Agent SFT            │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
 Human request ────►│ Deterministic Shell  │
                    │ + trained model      │
                    └──────────┬───────────┘
                               │
                    Search / Read / Exec
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Synthetic World      │
                    │ Environment          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Deterministic Eval   │
                    └──────────┴───────────┘
```

---

## Tool Set

- **SEARCH**: Deterministic BM25 / lexical search with stable tie-breaking.
- **READ**: Line-addressed document reader returning `D{id}:L{line}` identifiers.
- **EXEC**: Sandboxed Python AST evaluator (whitelisted arithmetic, lists, dicts, `min`, `max`, `sum`, `len`, `sorted`; zero filesystem, networking, or imports).

---

## Developer Quickstart

```bash
# 1. Generate synthetic world & corpus
make synth-smoke

# 2. Train synthetic BPE tokenizer from scratch
make tokenizer-smoke

# 3. Pretrain base transformer from random weights
make pretrain-smoke

# 4. Supervised fine-tuning on structured agent trajectories
make agent-sft-smoke

# 5. Run deterministic evaluation harness & baselines
make eval-smoke

# Or run the entire pipeline end-to-end:
make poc

# Run automated tests:
make test
```
