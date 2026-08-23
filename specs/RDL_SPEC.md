# Retrieval Domain Language (RDL) & Host Observation Protocol (HOP) Specification

## 1. Overview & Core Philosophy

The **Retrieval Domain Language (RDL)** and its companion **Host Observation Protocol (HOP)** form a compact, deterministic, token-optimized communication standard designed for procedural neural models ($35\text{M} - 150\text{M}$ parameters).

### Core Architectural Invariants
1. **Procedural Action vs. Contingent Knowledge**: The neural policy generates pure procedural steps (in-context pointer dereferencing, AST arithmetic, evidence citations). Contingent world knowledge resides exclusively in external tools and document stores.
2. **Zero Syntax Overhead via 1-to-1 Token Mapping**: Every opcode and protocol keyword maps directly to a discrete single token in the tokenizer vocabulary (`SEARCH`, `READ`, `FILTER`, `MATH`, `EMIT`, `ABSTAIN`, `OBS`, `LINES`, `LIMIT`, `EVIDENCE`, `REASON`). This eliminates multiline JSON serialization boilerplate (saving 60–75% of context tokens per turn) and prevents JSON delimiter crashes.
3. **Strict Deterministic Grammars ($LL(1)$ with 0 Lookahead)**: String literals for search queries and entity names are enclosed in double quotes (`"..."`), eliminating shift/reduce keyword collisions.
4. **Pure Numeric Arithmetic Safety**: The `MATH` opcode evaluates safe, constant-time infix numeric expressions with zero allowed identifiers or variables.
5. **First-Class Epistemic Provenance & Abstention**: Final assertions require verifiable line-level citations (`EMIT <ans> EVIDENCE [<doc_id>:<line>, ...]`) scored against the hidden `ProofGraph` in `eval/metrics.py`. Held-out missing evidence queries must be met with explicit `ABSTAIN` primitives.

---

## 2. Reconciled RDL + HOP Instruction Set

| Category | Opcode / Tag | Syntax | Purpose |
|---|---|---|---|
| **Action** | `SEARCH` | `SEARCH "<query>" [LIMIT <k>]` | Deterministic BM25 index query |
| **Action** | `READ` | `READ <doc_id> [LINES <start>-<end>]` | Line-addressed document fetch |
| **Action** | `FILTER` | `FILTER <field> <EQ\|GT\|LT\|CONTAINS> <literal>` | In-memory candidate filtering |
| **Action** | `MATH` | `MATH <infix_numeric_expr>` | Safe, pure numeric AST evaluation |
| **Action** | `EMIT` | `EMIT <ans> EVIDENCE [<doc_id>:<line>, ...]` | Grounded final answer with proof citations |
| **Action** | `ABSTAIN` | `ABSTAIN [REASON <reason>]` | Epistemic abstention on missing/conflicting data |
| **Host** | `OBS` | `OBS <SEARCH\|READ\|MATH\|ERROR> <payload>` | Token-optimized host observation return |

---

## 3. RDL Formal Grammar (EBNF)

```ebnf
(* Top-level turn emitted by the model *)
rdl_turn        ::= action_stmt | final_stmt

(* Action Statements: Tool Invocations *)
action_stmt     ::= search_stmt
                  | read_stmt
                  | filter_stmt
                  | math_stmt

search_stmt     ::= "SEARCH" string_literal ("LIMIT" integer)?
read_stmt       ::= "READ" doc_id ("LINES" integer "-" integer)?
filter_stmt     ::= "FILTER" identifier rel_op literal
math_stmt       ::= "MATH" math_expr

(* Safe Pure Numeric Math Grammar *)
math_expr       ::= math_term (("+" | "-") math_term)*
math_term       ::= math_factor (("*" | "/" | "//" | "%" | "^") math_factor)*
math_factor     ::= ("+" | "-")? number | "(" math_expr ")"

(* Final Statements: Termination & Provenance *)
final_stmt      ::= emit_stmt | abstain_stmt
emit_stmt       ::= "EMIT" answer_literal "EVIDENCE" "[" citation_list "]"
abstain_stmt    ::= "ABSTAIN" ("REASON" abstain_reason)?

(* Evidence Provenance Citations *)
citation_list   ::= citation ("," citation)*
citation        ::= doc_id ":" line_number
doc_id          ::= "D" [0-9]+
line_number     ::= [0-9]+

(* Literals & Primitives *)
string_literal  ::= '"' [^"\r\n]* '"'
answer_literal  ::= string_literal | number | identifier
number          ::= [0-9]+ ("." [0-9]+)?
integer         ::= [0-9]+
identifier      ::= [a-zA-Z_][a-zA-Z0-9_]*
rel_op          ::= "EQ" | "GT" | "LT" | "CONTAINS"

(* Abstention Reasons *)
abstain_reason  ::= "insufficient_evidence"
                  | "no_evidence"
                  | "conflict"
                  | "out_of_domain"
```

---

## 4. Host Observation Protocol (HOP)

The **Host Observation Protocol (HOP)** defines the deterministic, token-efficient format used by the execution environment to return tool observations to the model:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       HOST OBSERVATION PROTOCOL (HOP)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. SEARCH OBSERVATION                                                       │
│    OBS SEARCH [D01 (8.4), D04 (5.2)]       (Hits with BM25 scores)          │
│    OBS SEARCH EMPTY                        (Zero hits)                      │
│                                                                             │
│ 2. READ OBSERVATION                                                         │
│    OBS READ D01 LINES 1-3                  (Preserves D:L line numbering)   │
│    D01:L1 Census report for Valerius Fort.                                  │
│    D01:L2 Fort garrison count is 140.                                       │
│    D01:L3 Stationed in Northern valley.                                     │
│    OBS READ D01 NOT_FOUND                  (Invalid document ID)            │
│                                                                             │
│ 3. MATH OBSERVATION                                                         │
│    OBS MATH 420                            (Numeric result)                 │
│    OBS MATH ERROR <DIVISION_BY_ZERO | OVERFLOW | SYNTAX_ERROR>              │
│                                                                             │
│ 4. PROTOCOL ERROR / RESOURCE LIMIT                                          │
│    OBS ERROR <INVALID_OPCODE | RESOURCE_LIMIT_EXCEEDED | UNKNOWN_ARG>       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### HOP Formal Grammar (EBNF)
```ebnf
observation      ::= "OBS" (search_obs | read_obs | math_obs | error_obs)
search_obs       ::= "SEARCH" ("[" doc_hit ("," doc_hit)* "]" | "EMPTY")
doc_hit          ::= doc_id "(" score_val ")"
score_val        ::= [0-9]+ ("." [0-9]+)?
read_obs         ::= "READ" doc_id "LINES" integer "-" integer "\n" (line_entry "\n")+
                   | "READ" doc_id "NOT_FOUND"
line_entry       ::= doc_id ":L" integer " " [^\r\n]+
math_obs         ::= "MATH" (number | "ERROR" error_code)
error_obs        ::= "ERROR" error_code (" " string_literal)?
error_code       ::= "DIVISION_BY_ZERO"
                   | "OVERFLOW"
                   | "SYNTAX_ERROR"
                   | "INVALID_OPCODE"
                   | "RESOURCE_LIMIT_EXCEEDED"
                   | "UNKNOWN_ARG"
```

---

## 5. End-to-End Execution Trace Example

```text
[Episode Start]
<USER> What is the combined garrison of Fort Valerius and Fort Albia?
<ACTION> SEARCH "Fort Valerius" LIMIT 2
<OBSERVATION> OBS SEARCH [D01 (7.8), D04 (2.1)]
<ACTION> READ D01 LINES 1-2
<OBSERVATION> OBS READ D01 LINES 1-2
D01:L1 Fort Valerius registry record.
D01:L2 Active infantry stationed: 140.
<ACTION> SEARCH "Fort Albia" LIMIT 2
<OBSERVATION> OBS SEARCH [D08 (8.1)]
<ACTION> READ D08 LINES 1-2
<OBSERVATION> OBS READ D08 LINES 1-2
D08:L1 Fort Albia garrison report.
D08:L2 Total standing troops: 210.
<ACTION> MATH 140 + 210
<OBSERVATION> OBS MATH 350
<ACTION> EMIT "350" EVIDENCE [D01:2, D08:2]
[Episode Terminated - 100% Grounded]
```

---

## 6. Universal Multi-Modal Adapter Interface (`EvidenceProvider`)

```rust
pub trait EvidenceProvider: Send + Sync {
    /// Returns self-describing metadata and capabilities of the adapter
    fn introspect(&self) -> AdapterMetadata;

    /// Executes lexical BM25 or keyword search over documents or records
    fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>, AdapterError>;

    /// Reads line-addressed content or full document slices
    fn read(&self, doc_id: &str, lines: Option<LineRange>) -> Result<DocumentSlice, AdapterError>;

    /// Executes relational or structured filter queries
    fn filter(&self, predicate: Predicate) -> Result<Vec<Record>, AdapterError>;

    /// Executes dense vector semantic retrieval
    fn vector_search(&self, embedding: &[f32], k: usize) -> Result<Vec<SearchResult>, AdapterError>;
}
```

---

## 7. Deterministic $O(1)$ Trajectory Compilation from `ProofGraph`

Training trajectories for SFT and pretraining are compiled directly from the world generator's `ProofGraph` in $O(1)$ time per task, yielding canonical RDL actions and HOP observations without exponential search.

---

## 8. High-Throughput Host Runtime Architecture

- **In-Process Dispatch**: The host interpreter runs in-process with zero subprocess overhead, enabling $>2{,}000\text{ steps/sec}$ for RL/GRPO rollouts.
- **Strict AST Sandboxing**: `MATH` operations are restricted to pure numeric binary and unary arithmetic, preventing code execution or namespace leakage.
