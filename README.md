# Large Code Model (LCM) — Synthetic-Only Agentic Language Model POC

An end-to-end proof-of-concept for an alternative language-model architecture: a small neural model trained exclusively from random weights on deliberately constructed synthetic language that learns linguistic, semantic, planning, tool-use, and code-synthesis capabilities to solve novel factual tasks inside a deterministic agentic runtime rather than relying on parametric factual memorization.

---

## 1. Core Thesis

Conventional language models simultaneously encode language reasoning patterns and enormous amounts of contingent factual knowledge. This makes reasoning and memorization difficult to separate, leading to stale data, hallucinations, ungrounded answers, and bloated parameter sizes.

**LCM investigates a different decomposition:**
- **Neural Model**: Responsible for intent interpretation, planning, tool selection, query generation, program synthesis, observation interpretation, and final answer composition.
- **Deterministic Shell**: Responsible for execution state, tool invocations, permissions, sandboxing, iteration limits, evidence tracking, and protocol validation.
- **External Environment**: Responsible for contingent facts, documents, and world state.

The fundamental hypothesis is: **capacity devoted to memorizing contingent world facts can instead be devoted to learning *how to find out*.**

---

## 2. Knowledge Boundary

| Tier | Category | Status | Examples |
|---|---|---|---|
| **Tier A** | Linguistic & Semantic Primitives | Allowed in weights | Syntax, negation, conjunction, relative clauses, pronouns, spatial relations (*above/below, north/south*), quantitative terms (*more/fewer, larger/smaller*). |
| **Tier B** | Formal Invariants | Allowed in weights | Equality, transitivity, ordering, arithmetic (*add/sub/mul/div*), Boolean logic (*AND/OR/NOT*), set operations. |
| **Tier C** | Stable Physical Priors | Configurable in weights | Object permanence, non-co-location, containment hierarchies, event sequence. |
| **Tier D** | Contingent World Knowledge | **Strictly Externalized** | Historical dates, populations, geographic coordinates, prices, names of real people/companies, API endpoints. |

---

## 3. System Architecture

```
                    ┌────────────────────────┐
                    │  Synthetic World Spec  │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │     World / Corpus     │
                    │       Generator        │
                    └───────────┬────────────┘
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
                    │ Agent Trajectory SFT │
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
                    │ Synthetic World Env  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Deterministic Eval  │
                    │ + Proof-Graph Checks │
                    └──────────────────────┘
```

---

## 4. Deterministic Tool Set

1. **`SEARCH(query, limit)`**: Deterministic BM25 / lexical search over synthetic documents with stable tie-breaking.
2. **`READ(document_id)`**: Line-addressed document reader returning explicit `D{id}:L{line}` identifiers.
3. **`EXEC(code, inputs)`**: AST-restricted Python evaluator supporting safe arithmetic, comparisons, lists, dicts, `min`, `max`, `sum`, `len`, `sorted`, and list comprehensions (strictly zero filesystem, networking, imports, or dynamic reflection).

---

## 5. Epistemic Scoring Enforcement

For tasks labeled `REQUIRES_RETRIEVAL`:
- Producing the correct answer string **without** citing valid document line evidence from the hidden ground-truth `ProofGraph` is scored as an **`UNSUPPORTED_CLAIM` failure**.
- This prevents parametric lucky guesses from inflating evaluation scores.

---

## 6. Installation & GPU Setup (NVIDIA RTX 4090 / CUDA)

### Step 1: Clone Repository
```bash
git clone https://github.com/lbsa71/LCM.git
cd LCM
```

### Step 2: Set Up Virtual Environment & Dependencies
```bash
# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# For GPU machines (CUDA 12.1+ / RTX 4090):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CPU-only machines:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies
pip install tokenizers pydantic pyyaml pytest
```

### Step 3: Verify GPU Detection
```bash
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 7. Running Experiments

### Experiment Presets

| Preset | Model Size | Tokens | Purpose | Command |
|---|---|---|---|---|
| **`smoke`** | ~1M params | ~5M tokens | Fast pipeline debugging & testing | `make poc` or `--config configs/smoke.yaml` |
| **`small`** | ~30M params | ~50M tokens | Early learning curve | `--config configs/small.yaml` |
| **`primary`** | ~80M params | ~100–200M tokens | Main research POC benchmark | `--config configs/primary.yaml` |
| **`stretch`** | ~200M params | ~300M–1B tokens | Scaling experiment | `--config configs/stretch.yaml` |

### Step-by-Step CLI Execution

```bash
# 1. Generate synthetic world, documents, tasks, and trajectories
python3 -m synth.generate --config configs/primary.yaml

# 2. Train BPE tokenizer from scratch on synthetic corpus
python3 -m training.tokenizer --config configs/primary.yaml

# 3. Pretrain base transformer from random weights
python3 -m training.pretrain --config configs/primary.yaml

# 4. Supervised fine-tuning on structured agent trajectories
python3 -m training.agent_sft --config configs/primary.yaml

# 5. Run deterministic evaluation harness & comparative baselines
python3 -m eval.runner --config configs/primary.yaml
```

### Quick Execution via Makefile
```bash
# Full end-to-end smoke pipeline:
make poc

# Run automated test suite:
make test
```

---

## 8. Benchmark Evaluation Suites

1. **Suite A — Language Understanding**: Syntax, negation, relative clauses, reference resolution.
2. **Suite B — Invariant Reasoning**: Spatial ordering, temporal sequences, formal arithmetic invariants.
3. **Suite C — Single-Hop Retrieval**: Fact acquisition via `SEARCH` → `READ` → `FINAL`.
4. **Suite D — Multi-Hop Retrieval**: Composed relations (e.g. identify all members in region → retrieve populations → compare).
5. **Suite E — Retrieval + Computation**: Retrieval composed with `EXEC` code evaluation.
6. **Suite F — Missing Evidence**: Epistemic abstention (`insufficient_evidence`) when information is withheld.
7. **Suite G — Tool Recovery**: Recovery from injected tool errors and missed search queries.
8. **Anti-Memorization Suite**:
   - **World Permutation Test**: Evaluates identical query semantics across randomized worlds.
   - **Prior Reversal Test**: Reverses frequently seen correlations to measure evidence obedience.
   - **Evidence-Disabled Test**: Withholds critical documents to verify the agent abstains instead of guessing.
   - **Closed-Book Leakage Probe**: Queries real-world historical/geographic facts; correctly scores ungrounded answers as leakage.

---

## 9. Automated Test Suite

```bash
pytest tests/ -v
```

Unit and integration tests cover:
- World generator seed determinism & fact permutation
- Corpus linter forbidden entity detection
- Deterministic BM25 search scoring and tie-breaking
- Line-addressed document reader
- Restricted AST execution sandbox safety
- Protocol message validation and turn budgets
- Oracle 100% ground-truth baseline
- Epistemic ungrounded guess rejection

---

## 10. Repository Structure

```
.
├── configs/                  # Experiment presets (smoke, small, primary, stretch)
│   ├── smoke.yaml
│   ├── small.yaml
│   ├── primary.yaml
│   └── stretch.yaml
├── specs/                    # Formal boundaries, lexicons, and denylists
│   ├── knowledge_boundary.yaml
│   ├── lexicon.yaml
│   └── forbidden_entities.txt
├── synth/                    # Synthetic generation engine
│   ├── ontology.py           # Dataclasses (World, Entity, Fact, Document, ProofGraph)
│   ├── world.py              # Procedural world generator
│   ├── language/             # Grammar, lexicons, counterfactual pairs
│   ├── documents/            # Line-addressed document generator (D_id:L_line)
│   ├── tasks/                # Benchmark suites (A-G, anti-memorization)
│   ├── trajectories/         # Agent trajectory generator with loss masking
│   ├── lint.py               # Contamination and balance linter
│   ├── manifest.py           # Cryptographic SHA-256 manifest
│   └── generate.py           # Top-level synthetic CLI
├── training/                 # Model architecture & training pipelines
│   ├── model.py              # Pure PyTorch Decoder Transformer (RoPE, SwiGLU, RMSNorm)
│   ├── tokenizer.py          # Fast BPE tokenizer trainer from scratch
│   ├── pretrain.py           # Causal next-token pretraining
│   └── agent_sft.py          # Masked trajectory SFT trainer
├── agent/                    # Runtime deterministic shell & sandboxed tools
│   ├── protocol.py           # JSON schema models (PLAN, TOOL_CALL, FINAL)
│   ├── state.py              # Agent turn and resource tracking
│   ├── shell.py              # Deterministic ReAct loop
│   └── tools/                # Deterministic tools (search, read, exec)
├── eval/                     # Evaluation harness & comparative baselines
│   ├── oracle.py             # Ground-truth oracle solver (100% target)
│   ├── baselines/            # Majority, BoW, No-Tool, Rule-based
│   ├── metrics.py            # Epistemic metrics & failure taxonomy
│   └── runner.py             # Evaluation runner & HTML report generator
├── tests/                    # Unit and integration pytest suite
├── Makefile                  # Developer CLI targets
└── pyproject.toml            # Project packaging specification
```
