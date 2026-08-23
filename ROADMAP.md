# LCM Project Roadmap & Architecture Specification

## 1. Executive Summary & Core Philosophy

The **Language & Computation Model (LCM)** project develops compact, high-speed neural architectures ($35\text{M} - 150\text{M}$ parameters) specialized in **deterministic procedural execution, multi-hop reasoning, and zero hallucination**.

### Core Invariants
1. **Separation of Concerns**: Procedural execution (attention-routed token operations) is strictly divorced from contingent world knowledge (which resides exclusively in external tools and documents).
2. **Deterministic Grounding**: Every final assertion must be backed by verifiable document citations retrieved during the episode rollout.
3. **Information-Theoretic Induction**: Models are trained on infinite-lexicon synthetic environments to mathematically eliminate unigram memorization and enforce in-context copy heads.

---

## 2. Tool Architecture: Retrieval Domain Language (RDL) & Host Observation Protocol (HOP)

For the complete formal EBNF grammar, token mapping, and multi-modal adapter specification, see the formal specification document: [specs/RDL_SPEC.md](file:///c:/Users/lbsa7/Documents/Source/lbsa71/LCM/specs/RDL_SPEC.md).

### The Case for a Dedicated Domain Language
Standard general-purpose tools (Python REPL, raw SQL, Bash) present major structural drawbacks for compact procedural models:
- **Syntax Entropy**: Multiline indentation, string quoting escapes, and complex keywords consume token bandwidth and create fragile failure modes.
- **Ambiguous Tokenization**: Byte-pair tokenizers fragment keywords (`SELECT`, `WHERE`, `import`, `print`) across multiple subword pieces.
- **Execution Overhead**: Subprocess forks for Python/Bash introduce 50–100ms latency per step.

### Specification: Retrieval Domain Language (RDL)
Instead of arbitrary programming languages, LCM operates over an unambiguous, single-token-mapped **Retrieval Domain Language (RDL)**:

```
[RDL Instruction Set]
SEARCH   <query_string> [LIMIT <k>]
READ     <doc_id> [LINES <start>-<end>]
FILTER   <field> <OP: EQ|GT|LT|CONTAINS> <literal>
MATH     <infix_arithmetic_expr>
EMIT     <answer> EVIDENCE [<doc_id>:<line_num>, ...]
ABSTAIN  [REASON <no_evidence|conflict>]
```

### Architectural Advantages of RDL
1. **1-to-1 Token Mapping**: Every opcode (`SEARCH`, `READ`, `FILTER`, `MATH`, `EMIT`, `ABSTAIN`) corresponds to a dedicated single token ID in the tokenizer vocabulary.
2. **Zero Escaping or Quoting Crashes**: Argument slots have deterministic grammars parsed directly by the execution runtime.
3. **Extreme Throughput**: Native C++/Rust/Python host execution processes RDL operations in $<0.5\text{ ms}$.

---

## 3. Phased Milestone Roadmap

```
LCM Milestone Roadmap
├── Milestone 1: Pure In-Context Induction Proof of Concept (Current Phase)
│      ├── High-diversity combinatorial entity lexicon (>500k nonce permutations)
│      ├── Explicit in-context pointer & arithmetic induction pretraining
│      ├── Atomic state transitions with supervised <EOS> delimiters
│      └── Target: >90% Grounded Accuracy across synthetic evaluation suites (A–G)
│
├── Milestone 2: RDL Tool Formalization & Real Domain Schema Adapters (Month 1)
│      ├── Formal RDL bytecode specification and host runtime interpreter
│      ├── Adapters converting enterprise data sources (REST APIs, SQL schemas, vector indices) into RDL tables
│      └── Target: Outperform GPT-4o-mini on complex multi-hop enterprise retrieval benchmarks
│
├── Milestone 3: Trajectory Optimization & Active Self-Correction (Month 2)
│      ├── Direct Preference Optimization (DPO) and Reinforcement Learning (PPO/GRPO) on agent trajectories
│      ├── Dynamic error recovery (backtracking when search queries return 0 hits)
│      └── Target: 99.9% protocol compliance under adversarial distractors
│
└── Milestone 4: Edge Distillation & Ultra-Low Latency Sandboxing (Month 3)
       ├── 4-bit / 8-bit quantization (GGUF, ONNX, WebGPU)
       └── Sub-10ms procedural inference running locally in memory-constrained environments
```

---

## 4. Compute Scaling & Hardware Economics

| Milestone | Parameter Scale | Target Hardware | Est. Training Time | Wall-Clock Cost |
| :--- | :--- | :--- | :--- | :--- |
| **M1: Synthetic POC** | $35\text{M} - 70\text{M}$ | $1\times \text{RTX 4090 (16GB)}$ | 1.5 – 2 hours | **\$0 (Local)** |
| **M2: RDL Domain Models** | $70\text{M} - 150\text{M}$ | $1\times \text{RTX 4090 (16GB)}$ | 3 – 4 hours | **\$0 (Local)** |
| **M3: DPO / RL Scaling** | $150\text{M} - 350\text{M}$ | $4\times \text{A100 / H100}$ | 1 – 2 hours | $\approx \$80 - \$150$ |
| **M4: Production Flagship** | $500\text{M} - 1\text{B}$ | $8\times \text{H100}$ | 2 – 4 hours | $\approx \$300 - \$600$ |

*Conclusion*: The local RTX 4090 GPU is fully sufficient to deliver functional proof of concept through Milestones 1 and 2 without requiring external cloud expenditure.
