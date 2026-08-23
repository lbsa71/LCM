"""Unit tests for Retrieval Domain Language (RDL) tokenization, protocol parsing, and Host Observation Protocol (HOP)."""

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from agent.protocol import (
    ToolCallMessage,
    FinalMessage,
    ProtocolError,
    parse_and_validate_message,
    parse_rdl_message,
    format_search_hop,
    format_read_hop,
    format_math_hop,
    format_error_hop,
)
from agent.tools.exec import RestrictedASTEvaluator
from training.tokenizer import SPECIAL_TOKENS
from synth.ontology import Document, DocumentLine


def test_rdl_and_hop_special_tokens_defined():
    """Ensure all required role delimiters are in SPECIAL_TOKENS."""
    required_tokens = [
        "<PAD>", "<BOS>", "<EOS>", "<UNK>", "<USER>", "<ASSISTANT>",
        "<TOOL>", "<OBSERVATION>", "<PLAN>", "<ACTION>", "<FINAL>"
    ]
    for tok in required_tokens:
        assert tok in SPECIAL_TOKENS, f"Token '{tok}' missing from SPECIAL_TOKENS"


def test_rdl_single_token_mapping():
    """Ensure RDL and HOP opcodes are cleanly tokenized."""
    from tokenizers import decoders
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=1000,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    training_data = [
        "SEARCH READ FILTER MATH EMIT ABSTAIN OBS EVIDENCE REASON EMPTY NOT_FOUND ERROR",
        " SEARCH READ FILTER MATH EMIT ABSTAIN OBS EVIDENCE REASON EMPTY NOT_FOUND ERROR"
    ] * 20
    tokenizer.train_from_iterator(training_data, trainer)

    # Verify RDL / HOP opcodes decode cleanly
    for op in ["SEARCH", "READ", "FILTER", "MATH", "EMIT", "ABSTAIN"]:
        encoded = tokenizer.encode(op)
        decoded = tokenizer.decode(encoded.ids)
        assert decoded == op


def test_parse_rdl_search_statement():
    """Verify SEARCH statement parsing with and without LIMIT and with strict string literals."""
    msg1 = parse_and_validate_message('SEARCH "Valerius Fort" LIMIT 3')
    assert isinstance(msg1, ToolCallMessage)
    assert msg1.tool == "search"
    assert msg1.arguments == {"query": "Valerius Fort", "limit": 3}

    msg2 = parse_and_validate_message('SEARCH "Aurelius Fortress"')
    assert isinstance(msg2, ToolCallMessage)
    assert msg2.tool == "search"
    assert msg2.arguments == {"query": "Aurelius Fortress", "limit": 5}

    # Backward compatibility with bare words
    msg3 = parse_and_validate_message('SEARCH barel LIMIT 3')
    assert isinstance(msg3, ToolCallMessage)
    assert msg3.tool == "search"
    assert msg3.arguments == {"query": "barel", "limit": 3}


def test_parse_rdl_read_statement():
    """Verify READ statement parsing with and without line ranges."""
    msg1 = parse_and_validate_message("READ D01")
    assert isinstance(msg1, ToolCallMessage)
    assert msg1.tool == "read"
    assert msg1.arguments == {"document_id": "D01"}

    msg2 = parse_and_validate_message("READ D02 LINES 3-7")
    assert isinstance(msg2, ToolCallMessage)
    assert msg2.tool == "read"
    assert msg2.arguments == {"document_id": "D02", "lines": [3, 4, 5, 6, 7]}


def test_parse_rdl_math_statement():
    """Verify MATH statement parsing."""
    msg = parse_and_validate_message("MATH 140 + 260")
    assert isinstance(msg, ToolCallMessage)
    assert msg.tool == "exec"
    assert msg.arguments == {"code": "140 + 260"}


def test_parse_rdl_emit_with_evidence_provenance():
    """Verify EMIT statement with line-level evidence citations."""
    msg1 = parse_and_validate_message("EMIT 613 EVIDENCE [D01:2]")
    assert isinstance(msg1, FinalMessage)
    assert msg1.answer == "613"
    assert len(msg1.evidence) == 1
    assert msg1.evidence[0].document_id == "D01"
    assert msg1.evidence[0].lines == [2]

    msg2 = parse_and_validate_message('EMIT "Silver Key" EVIDENCE [D01:2, D04:12, D04:13]')
    assert isinstance(msg2, FinalMessage)
    assert msg2.answer == "Silver Key"
    assert len(msg2.evidence) == 2
    assert msg2.evidence[0].document_id == "D01"
    assert msg2.evidence[0].lines == [2]
    assert msg2.evidence[1].document_id == "D04"
    assert msg2.evidence[1].lines == [12, 13]


def test_parse_rdl_abstain_statement():
    """Verify ABSTAIN statement parsing."""
    msg1 = parse_and_validate_message("ABSTAIN")
    assert isinstance(msg1, FinalMessage)
    assert msg1.answer == "insufficient_evidence"
    assert len(msg1.evidence) == 0

    msg2 = parse_and_validate_message("ABSTAIN REASON insufficient_evidence")
    assert isinstance(msg2, FinalMessage)
    assert msg2.answer == "insufficient_evidence"
    assert len(msg2.evidence) == 0

    msg3 = parse_and_validate_message("ABSTAIN REASON conflict")
    assert isinstance(msg3, FinalMessage)
    assert msg3.answer == "conflict"
    assert len(msg3.evidence) == 0


def test_hop_search_formatting():
    """Verify Host Observation Protocol formatting for search."""
    hits = [
        {"document_id": "D01", "score": 8.4},
        {"document_id": "D04", "score": 5.2}
    ]
    obs1 = format_search_hop(hits)
    assert obs1 == "OBS SEARCH [D01 (8.4), D04 (5.2)]"

    obs_empty = format_search_hop([])
    assert obs_empty == "OBS SEARCH EMPTY"


def test_hop_read_formatting():
    """Verify Host Observation Protocol formatting for read."""
    doc = Document(
        id="D01",
        title="Census report for Valerius Fort.",
        doc_type="census",
        lines=[
            DocumentLine(line_number=1, text="Census report for Valerius Fort.", fact_ids=[]),
            DocumentLine(line_number=2, text="Fort garrison count is 140.", fact_ids=[]),
            DocumentLine(line_number=3, text="Stationed in Northern valley.", fact_ids=[]),
        ]
    )
    obs_lines = format_read_hop(doc, lines=[1, 2, 3])
    expected = "OBS READ D01 LINES 1-3\nD01:L1 Census report for Valerius Fort.\nD01:L2 Fort garrison count is 140.\nD01:L3 Stationed in Northern valley."
    assert obs_lines == expected

    obs_not_found = format_read_hop(None, doc_id="D99")
    assert obs_not_found == "OBS READ D99 NOT_FOUND"


def test_hop_math_formatting():
    """Verify Host Observation Protocol formatting for math and errors."""
    obs1 = format_math_hop(420)
    assert obs1 == "OBS MATH 420"

    obs2 = format_math_hop(350.5)
    assert obs2 == "OBS MATH 350.5"

    obs_err = format_math_hop(None, error="DIVISION_BY_ZERO")
    assert obs_err == "OBS MATH ERROR DIVISION_BY_ZERO"

    obs_proto_err = format_error_hop("RESOURCE_LIMIT_EXCEEDED")
    assert obs_proto_err == "OBS ERROR RESOURCE_LIMIT_EXCEEDED"


def test_pure_math_evaluator_safety():
    """Verify RestrictedASTEvaluator pure arithmetic evaluation and safety boundaries."""
    evaluator = RestrictedASTEvaluator()
    
    # Valid pure arithmetic
    res1 = evaluator.evaluate_pure_math("(140 + 280) * 1.05")
    assert res1["status"] == "success"
    assert abs(res1["result"] - 441.0) < 1e-6

    res2 = evaluator.evaluate_pure_math("2 ^ 8")
    assert res2["status"] == "success"
    assert res2["result"] == 256

    res3 = evaluator.evaluate_pure_math("10 // 3 + 10 % 3")
    assert res3["status"] == "success"
    assert res3["result"] == 4

    # Rejection of identifiers and variables
    res_id = evaluator.evaluate_pure_math("x + 10")
    assert res_id["status"] == "error"
    assert res_id["error_type"] in ("SYNTAX_ERROR", "RUNTIME_ERROR", "SECURITY_VIOLATION")

    # Rejection of function calls
    res_fn = evaluator.evaluate_pure_math("min(5, 10)")
    assert res_fn["status"] == "error"
