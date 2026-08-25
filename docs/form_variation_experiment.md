# Semantic Form-Variation Pilot

**Date:** 2026-08-25  
**Status:** Complete pilot; supports a separate semantic-parser experiment family.

## Question

How much controlled variation in wording is required before a scratch-trained
parser recognizes an operation independently of its surface form?

## Design

The experiment trains a small decoder-only parser from scratch to emit one of
three canonical semantic targets: `OP=ADD`, `OP=SUBTRACT`, or `OP=COMPARE`.
All conditions receive 400 full-batch training updates (819,200 padded input
tokens per seed). The only manipulated factor is the number of language
templates available per operation: 1, 2, 4, or 8.

Each condition uses two seeds. Evaluation separates exact training forms,
held-out forms for the same number pairs, unseen number pairs expressed in
seen forms, and three closely worded contrast prompts.

## Results

| Templates per operation | Seen form | Held-out form | Unseen operands, seen form | Minimal contrasts |
|---:|---:|---:|---:|---:|
| 1 | 100.0% | 35.2% | 100.0% | 50.0% |
| 2 | 100.0% | 37.5% | 100.0% | 50.0% |
| 4 | 100.0% | 50.9% | 100.0% | 66.7% |
| 8 | 100.0% | 54.9% | 100.0% | 100.0% |

The parser perfectly learns the canonical operation for observed forms and
unseen operands in familiar forms. Held-out form accuracy improves by 19.7
percentage points between one and eight templates, but remains far from
form-invariant. Controlled wording diversity also improves sensitivity to
near-neighbour meanings, rather than merely encouraging a generic arithmetic
label.

## Interpretation

The result supports a two-layer LCM architecture: a semantic parser can learn
a stable canonical target, while the procedural controller can be trained on
the target rather than every wording variant. It also shows that template
diversity must be treated as an explicit scaling dimension, not as incidental
augmentation.

This pilot measures operation selection only; it does not yet evaluate argument
extraction, nested semantic frames, dialogue context, or an end-to-end ReAct
controller. Those are the next experiment classes, and can reuse the isolated
`eval.form_variation` split and reporting pattern without changing the existing
scratch or SmolLM pipelines.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m eval.form_variation --config configs/form_variation.yaml
```

Artifacts are written to `runs/form_variation/results.json`.
