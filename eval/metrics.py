"""Comprehensive benchmark metrics and failure taxonomy calculator (PRD Section 34, 46)."""

from typing import Any, Dict, List


def evaluate_episode_outcome(episode: Dict[str, Any], task: Any) -> Dict[str, Any]:
    """Scores an individual episode with strict epistemic and grounded proof graph enforcement."""
    if hasattr(task, "gold_answer"):
        gold_answer = str(getattr(task, "gold_answer", "")).strip().lower()
        is_retrieval_required = getattr(task, "is_retrieval_required", True)
        is_insufficient_evidence = getattr(task, "is_insufficient_evidence", False)
        pg = getattr(task, "proof_graph", None)
        required_evidence = pg.required_document_lines if pg else {}
        metadata = getattr(task, "metadata", {}) or {}
    else:
        gold_answer = str(task.get("gold_answer", "")).strip().lower()
        is_retrieval_required = task.get("is_retrieval_required", True)
        is_insufficient_evidence = task.get("is_insufficient_evidence", False)
        required_evidence = task.get("required_evidence", {})  # doc_id -> list of line numbers
        metadata = task.get("metadata", {}) or {}

    model_answer = str(episode.get("model_answer", "")).strip().lower()
    raw_match = (model_answer == gold_answer)
    prior_answer = str(metadata.get("prior_answer", "")).strip().lower() if metadata.get("prior_answer") else None
    prior_contaminated = (prior_answer is not None and model_answer == prior_answer)

    # Validate evidence
    cited_evidence = episode.get("cited_evidence", [])
    valid_lines_cited = 0
    total_required_lines = sum(len(lines) for lines in required_evidence.values())
    
    for cite in cited_evidence:
        d_id = cite.get("document_id")
        c_lines = cite.get("lines", [])
        if d_id in required_evidence:
            req_lines = required_evidence[d_id]
            for cl in c_lines:
                if cl in req_lines:
                    valid_lines_cited += 1

    evidence_valid = True
    if is_retrieval_required and not is_insufficient_evidence:
        # Epistemic enforcement: Retrieval-required tasks must cite valid evidence
        evidence_valid = (valid_lines_cited > 0)

    grounded_success = raw_match and evidence_valid

    # Failure taxonomy classification
    failure_category = None
    if not grounded_success:
        if prior_contaminated:
            failure_category = "PRIOR_CONTAMINATION_ERROR"
        elif episode.get("termination_reason") in ("MAX_TURNS_EXCEEDED", "MAX_SEARCH_CALLS_EXCEEDED"):
            failure_category = "STEP_LIMIT"
        elif is_insufficient_evidence and model_answer != "insufficient_evidence":
            failure_category = "FAILED_ABSTENTION"
        elif is_retrieval_required and raw_match and not evidence_valid:
            failure_category = "UNSUPPORTED_CLAIM"
        elif episode.get("read_count", 0) == 0 and is_retrieval_required:
            failure_category = "FAILED_RETRIEVAL"
        else:
            failure_category = "COMPUTATION_OR_EXTRACTION_ERROR"

    coverage = (valid_lines_cited / max(1, total_required_lines)) if total_required_lines > 0 else 1.0

    task_id = getattr(task, "task_id", task.get("task_id") if isinstance(task, dict) else "")
    suite = getattr(task, "suite", task.get("suite") if isinstance(task, dict) else "")

    return {
        "task_id": task_id,
        "suite": suite,
        "raw_match": raw_match,
        "evidence_valid": evidence_valid,
        "grounded_success": grounded_success,
        "prior_contaminated": prior_contaminated,
        "evidence_coverage": round(coverage, 4),
        "failure_category": failure_category,
        "turns_used": episode.get("turns_used", 0)
    }


def compute_aggregate_metrics(eval_outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates episode outcomes across suites and computes benchmark summary."""
    total = len(eval_outcomes)
    if total == 0:
        return {}

    task_success_cnt = sum(1 for o in eval_outcomes if o["raw_match"])
    grounded_success_cnt = sum(1 for o in eval_outcomes if o["grounded_success"])
    unsupported_claims_cnt = sum(1 for o in eval_outcomes if o["raw_match"] and not o["evidence_valid"])
    prior_contamination_cnt = sum(1 for o in eval_outcomes if o.get("prior_contaminated", False))

    # Per-suite breakdown
    suites: Dict[str, List[Dict[str, Any]]] = {}
    for o in eval_outcomes:
        s = o.get("suite", "unknown")
        if s not in suites:
            suites[s] = []
        suites[s].append(o)

    suite_metrics = {}
    for s_name, s_outcomes in suites.items():
        s_tot = len(s_outcomes)
        s_raw = sum(1 for o in s_outcomes if o["raw_match"])
        s_grd = sum(1 for o in s_outcomes if o["grounded_success"])
        s_pcr = sum(1 for o in s_outcomes if o.get("prior_contaminated", False))
        suite_metrics[s_name] = {
            "total_tasks": s_tot,
            "task_success_rate": round(s_raw / s_tot, 4),
            "grounded_success_rate": round(s_grd / s_tot, 4),
            "prior_contamination_rate": round(s_pcr / s_tot, 4)
        }

    # Failure breakdown
    failures: Dict[str, int] = {}
    for o in eval_outcomes:
        f_cat = o.get("failure_category")
        if f_cat:
            failures[f_cat] = failures.get(f_cat, 0) + 1

    return {
        "total_eval_tasks": total,
        "overall_task_success_rate": round(task_success_cnt / total, 4),
        "overall_grounded_success_rate": round(grounded_success_cnt / total, 4),
        "unsupported_claim_rate": round(unsupported_claims_cnt / total, 4),
        "overall_prior_contamination_rate": round(prior_contamination_cnt / total, 4),
        "suite_metrics": suite_metrics,
        "failure_taxonomy_distribution": failures
    }
