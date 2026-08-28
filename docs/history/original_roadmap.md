# Archived original LCM roadmap

> **Superseded historical document.** This was an aspirational plan, not a
> report of measured results. Its projected accuracies, timelines, “zero
> hallucination,” and product comparisons were not established by the research
> and must not be used as current claims or execution instructions. See the
> current [research synthesis](../../README.md) and deferred
> [kill-test protocol](../kill_test.md).

## Original text: LCM Project Roadmap & Architecture Specification

## 1. Executive Summary & Core Philosophy

The **Language & Computation Model (LCM)** project develops compact, high-speed neural architectures ($35\text{M} - 150\text{M}$ parameters) specialized in **deterministic procedural execution, multi-hop reasoning, and zero hallucination**.

### Core Invariants
1. **Separation of Concerns**: Procedural execution (attention-routed token operations) is strictly divorced from contingent world knowledge (which resides exclusively in external tools and documents).
2. **Deterministic Grounding**: Every final assertion must be backed by verifiable document citations retrieved during the episode rollout.
3. **Information-Theoretic Induction**: Models are trained on infinite-lexicon synthetic environments to mathematically eliminate unigram memorization and enforce in-context copy heads.

---

## 2. Tool Architecture: Retrieval Domain Language (RDL) & Host Observation Protocol (HOP)

For the original formal grammar and adapter proposal, see
[the RDL specification](../../specs/RDL_SPEC.md). That specification is a
historical design document, not evidence that every projected property was
implemented or measured.

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

## 4. Continuous Dataset Scaling & Hardware Feasibility

### A. Scaling Physics & Hardware Bounds (RTX 4090 24GB)

Training compute requirements are governed by standard Chinchilla/Transformer FLOP and memory models:

$$\text{Training FLOPs} \approx 6 \times P \times D$$
$$\text{Wall-Clock Time (Hours)} = \frac{6 \times P \times D}{\text{Effective TFLOPs/s} \times 10^{12} \times 3600}$$

* **RTX 4090 Effective Throughput**: $\approx 135\text{ TFLOPs/sec}$ (BF16 with FlashAttention-2 and mixed precision at $\approx 41\%$ Model FLOPs Utilization).
* **VRAM Footprint (AdamW + BF16 Gradients + Activations)**:
  $$\text{VRAM} \approx 16 \times P + \text{Activations (2–4 GB)}$$
  *(Reduced to $10 \times P$ using 8-bit AdamW / GaLore).*

---

### B. Progressive Dataset Scaling & Error Rate Trajectory

| Phase | Model Size ($P$) | Training Tokens ($D$) | Training FLOPs | VRAM Footprint | Wall-Clock on 1× RTX 4090 | RDL Syntax Error Rate | Multi-Hop Retrieval Accuracy | Epistemic Grounding (Zero Hallucination) | Feasibility on RTX 4090 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1: Smoke / POC** | $35\text{M}$ | $20\text{M}$ | $4.2 \times 10^{15}$ | $\approx 2.5\text{ GB}$ | **9 minutes** | $\approx 4.2\%$ | $72.0\%$ | $84.5\%$ | **Trivial** |
| **S2: Small v3** | $80\text{M}$ | $100\text{M}$ | $4.8 \times 10^{16}$ | $\approx 3.2\text{ GB}$ | **1.7 hours** | $\approx 1.1\%$ | $86.5\%$ | $93.0\%$ | **Trivial** |
| **S3: Overnight (Current)** | $300\text{M}$ | $1\text{B}$ | $1.8 \times 10^{18}$ | $\approx 7.5\text{ GB}$ | **3.7 hours** | $< 0.3\%$ | $93.8\%$ | $98.1\%$ | **Optimal** |
| **S4: Extended In-Context** | $300\text{M}$ | $10\text{B}$ | $1.8 \times 10^{19}$ | $\approx 7.5\text{ GB}$ | **37 hours** (1.5 days) | $< 0.05\%$ | $96.5\%$ | $99.4\%$ | **Very Practical** |
| **S5: Linguistic Hybrid Vocab** | $500\text{M}$ | $30\text{B}$ | $9.0 \times 10^{19}$ | $\approx 12.0\text{ GB}$ | **185 hours** (7.7 days) | $< 0.01\%$ | $98.2\%$ | $99.8\%$ | **Ceiling for single-GPU** |
| **S6: 4090 Hard Wall** | $750\text{M}$ | $75\text{B}$ | $3.4 \times 10^{20}$ | $\approx 18.5\text{ GB}$ | **695 hours** (29 days) | $< 0.005\%$ | $99.1\%$ | $> 99.9\%$ | **Infeasible (Too slow)** |
| **S7: Cluster Production** | $1.2\text{B}$ | $200\text{B}$ | $1.4 \times 10^{21}$ | $\approx 28.0\text{ GB}$ (OOM) | *N/A (OOM)*<br>*(48 hrs on 8× H100)* | $< 0.001\%$ | $> 99.7\%$ | $> 99.99\%$ | **Requires Multi-GPU Cluster** |

---

### C. The RTX 4090 Hardware Limits & Transition Points

The limitation on the RTX 4090 is **wall-clock iteration time rather than pure VRAM capacity**:

1. **The Time Infeasibility Threshold ($> 10\text{ days}$ per experiment)**:
   - **Boundary**: **$\approx 500\text{M}$ parameters on $30\text{B}$ tokens** (7.7 days).
   - Beyond $30\text{B}$ tokens, single-run experiment turnaround exceeds 2 weeks, making hyperparameter sweeps and debugging unfeasible on a single card.

2. **The VRAM OOM Threshold ($> 24\text{ GB}$)**:
   - **Boundary**: **$\approx 1.1\text{B}$ parameters** (standard AdamW at sequence length 2048).
   - Above 1.1B parameters, optimizer states and gradients exceed 24 GB, requiring ZeRO-Offload or multi-GPU pipeline parallelism.

3. **Inference vs Training Feasibility**:
   - For **serving and inference**, the RTX 4090 remains fully capable up to **7B–13B quantized models** ($<10\text{ ms}$ latency). The hard wall is strictly on the training side.

---

### D. Execution Sequence & Cloud Transition Milestone

```
[Phase 1] S3: 300M / 1B tokens (3.7 hrs on RTX 4090) ──► Validate baseline error convergence
   │
   ▼
[Phase 2] S4: 300M / 10B tokens (37 hrs on RTX 4090) ──► Test stability under 10x trajectory density
   │
   ▼
[Phase 3] S5: 500M / 30B tokens (7.7 days on RTX 4090) ──► Final single-GPU frontier (Full open vocabulary)
   │
   ▼
[Phase 4] S7: 1.2B / 200B tokens (48 hrs on 8× H100 Cluster) ──► Production Flagship
```

---

## 5. Empirical Runtime Benchmarks: Projected vs. Actual

Across all training experiments and model scales, automated wall-clock timers and per-step latency logging (`utils/timer.py`, `training_metrics.json`) have measured empirical hardware throughput against initial planning projections:

### A. Runtime Performance Ledger

| Experiment Preset & Model Scale | Phase / Task | Projected Runtime | Actual Wall-Clock Runtime | Throughput / Speedup Factor |
| :--- | :--- | :--- | :--- | :--- |
| **`smoke`** (1.2M params) | End-to-End Pipeline (Data + Pretrain + SFT + Eval) | ~1–2 min | **45.2 s** | **~2.0× faster** |
| **`small_v2`** (35.9M params) | Pretraining (2,000 steps, 65.5M tokens) | ~15–20 min | **10.9 min** (654.1s) | 100.2k tok/s (~1.5× faster) |
| | Agent SFT (1,000 steps, 16.4M tokens) | ~5–7 min | **2.3 min** (138.3s) | 118.5k tok/s (~2.5× faster) |
| | Evaluation (Baselines + Milestones) | ~3–5 min | **2.5 min** | 550 ms/task |
| | **`small_v2` Total** | **~25–30 min** | **15.7 min** | **~1.7× overall speedup** |
| **`small_v3`** (35.9M params + RDL) | Pretraining (2,000 steps, 65.5M tokens) | ~10–12 min | **9.9 min** (594.7s) | 110.2k tok/s |
| | Agent SFT (1,000 steps, 16.4M tokens) | ~2.5–3 min | **2.3 min** (137.6s) | 119.1k tok/s |
| | Evaluation & Trajectory Verification | ~3–4 min | **2.8 min** | 420 ms/task |
| | **`small_v3` Total** | **~16–20 min** | **15.0 min** | **~1.2× overall speedup** |
| **`scaled_overnight_100m`** (119.5M params) | Dataset Gen (4,400 worlds) | ~1.5–2 min | **1.2 min** (72.0s) | 61 worlds/sec |
| | Pretraining (6,000 steps, 49.2M tokens) | ~25–35 min | **18.1 min** (1,083.2s) | 45.4k tok/s (~1.6× faster) |
| | Agent SFT (3,500 steps, 28.7M tokens) | ~15–20 min | **10.3 min** (618.7s) | 46.3k tok/s (~1.7× faster) |
| | Milestone Scaling (5 checkpoints × 180 tasks) | ~10–15 min | **6.5 min** (390.2s) | 397 ms/task |
| | **`scaled_overnight_100m` Total** | **~55–75 min** | **36.1 min** | **~1.8× overall speedup** |
| **`scaled_overnight_300m`** (310.4M params) | Dataset Gen (4,400 worlds + Suites A–I) | ~1.5–2 min | **1.3 min** (78.0s) | Completed |
| | Pretraining (6,000 steps, batch 4×12, 2048 ctx) | ~60–85 min | *Active StepTimer Log* | Estimated ~40k tok/s |
| | Agent SFT (3,500 steps, batch 4×8) | ~35–50 min | *Scheduled* | Estimated ~42k tok/s |
| | Evaluation (Held-out Suites A–I) | ~15–20 min | *Scheduled* | Estimated ~400 ms/task |
| | **`scaled_overnight_300m` Total** | **~110–155 min** | *In Progress* | — |

---

### B. Architectural Drivers of High Throughput

1. **TF32 Matrix Core Optimization**: Enabling TF32 for matrix multiplications and convolutions (`torch.backends.cuda.matmul.allow_tf32 = True`) delivers sustained 45k–119k tokens/sec on mobile RTX 4090 silicon.
2. **RDL In-Process Interpreter Speedup**: Direct in-process execution of RDL opcodes eliminates subprocess serialization overhead, achieving sub-millisecond dispatch (<0.5 ms/turn) and cutting evaluation wall-clock time by ~60%.
3. **Single-Digit Tokenization Efficiency**: Eliminating number fragmentation via `pre_tokenizers.Digits(individual_digits=True)` reduces sequence lengths by 22%, directly improving effective token throughput per step.
