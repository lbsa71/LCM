"""Synthetic agent trajectory generator for SFT with closed-loop dynamic environment execution, atomic state transitions, RDL actions, and HOP observations."""

import json
import re
import random
from typing import Any, Dict, List, Optional
from synth.ontology import World, Task, Document
from agent.protocol import format_search_hop, format_read_hop, format_math_hop
from agent.adapters.document import DocumentEvidenceProvider


def extract_entity_name_from_question(question: str, world: World) -> str:
    """Deterministically extracts the most relevant entity name from the question."""
    lower_q = question.lower()
    best_match = None
    best_len = 0
    for ent in world.entities.values():
        if ent.name.lower() in lower_q:
            if len(ent.name) > best_len:
                best_match = ent.name
                best_len = len(ent.name)
    if best_match:
        return best_match
    
    # Fallback: regex search for capitalized words or quotes
    quoted = re.findall(r'["\']([^"\']+)["\']', question)
    if quoted:
        return quoted[0]
    words = [w.strip("?,.:;") for w in question.split() if w.strip("?,.:;").istitle()]
    if words:
        return words[-1]
    return "record"


def _format_evidence_str(req_doc_lines: Dict[str, List[int]]) -> str:
    """Formats required document lines into RDL evidence citation string."""
    cites = []
    for d_id, l_nos in req_doc_lines.items():
        for l_no in l_nos:
            cites.append(f"{d_id}:{l_no}")
    return f"[{', '.join(cites)}]"


def _ensure_doc_in_search_hits(hits: List[Dict[str, Any]], target_doc_id: str, default_score: float = 3.5) -> List[Dict[str, Any]]:
    """Ensures the target document ID is present in the search hits list."""
    if not any(h.get("document_id") == target_doc_id for h in hits):
        hits = [{"document_id": target_doc_id, "score": default_score}] + hits[:2]
    return hits


def _infer_target_attribute(question: str) -> str:
    """Infers target attribute name for precise in-context plan grounding."""
    q_low = question.lower()
    if "population" in q_low or "headcount" in q_low or "inhabitant" in q_low:
        return "population"
    elif "status" in q_low or "active" in q_low or "critical" in q_low or "calibrated" in q_low or "dormant" in q_low:
        return "status"
    elif "located" in q_low or "where" in q_low or "region" in q_low or "inside" in q_low:
        return "location"
    elif "measured" in q_low or "value" in q_low or "reading" in q_low:
        return "measured_value"
    return "answer"


class TrajectoryGenerator:
    """Produces closed-loop procedural ReAct agent trajectories with live environment search and HOP observations."""

    def __init__(self, include_plan: bool = False):
        self.include_plan = include_plan

    def generate_trajectory_for_task(self, world: World, task: Task, rng: random.Random) -> Dict[str, Any]:
        """Generates a complete structured trajectory for a given task using live world adapters."""
        doc_adapter = DocumentEvidenceProvider(world)
        turns = []
        
        # User message
        turns.append({
            "role": "user",
            "content": task.question,
            "train": False
        })

        if task.suite == "suite_h_direct_computation":
            # Direct computation (Math / String Ops): user -> MATH expr -> OBS MATH -> EMIT
            expr = task.proof_graph.goal.replace("direct_computation: ", "").strip() if task.proof_graph else ""
            if not expr:
                expr = task.gold_answer

            turns.append({"role": "action", "content": f"MATH {expr}", "train": True})
            turns.append({"role": "observation", "content": format_math_hop(task.gold_answer), "train": False})
            turns.append({"role": "final", "content": f'EMIT "{task.gold_answer}" EVIDENCE []', "train": True})

        elif not task.is_retrieval_required:
            if task.suite == "suite_b_invariants" and ("sum" in task.question.lower() or "plus" in task.question.lower() or "total" in task.question.lower()):
                # Suite B with arithmetic: user -> MATH -> OBS MATH -> EMIT
                nums = re.findall(r'\b\d+\b', task.question)
                if len(nums) >= 2:
                    expr = f"{nums[0]} + {nums[1]}"
                else:
                    expr = task.gold_answer

                try:
                    res_val = eval(expr)
                except Exception:
                    res_val = task.gold_answer

                turns.append({"role": "action", "content": f"MATH {expr}", "train": True})
                turns.append({"role": "observation", "content": format_math_hop(res_val), "train": False})
                turns.append({"role": "final", "content": f'EMIT "{res_val}" EVIDENCE []', "train": True})

            else:
                # Direct invariant / language task (Suite A): user -> EMIT
                turns.append({
                    "role": "final",
                    "content": f'EMIT "{task.gold_answer}" EVIDENCE []',
                    "train": True
                })

        elif task.is_insufficient_evidence:
            # Missing evidence trajectory: user -> PLAN -> SEARCH -> OBS SEARCH EMPTY -> ABSTAIN
            query_entity = extract_entity_name_from_question(task.question, world)

            turns.append({"role": "plan", "content": f'PLAN GOAL: retrieve("{query_entity}"); SEARCH "{query_entity}"', "train": True})
            turns.append({"role": "action", "content": f'SEARCH "{query_entity}" LIMIT 3', "train": True})
            turns.append({"role": "observation", "content": format_search_hop([]), "train": False})
            turns.append({"role": "plan", "content": "PLAN NO_EVIDENCE_FOUND; ACTION: ABSTAIN", "train": True})
            turns.append({"role": "final", "content": "ABSTAIN REASON insufficient_evidence", "train": True})

        elif task.suite == "suite_e_retrieval_computation":
            # Multi-document or single-document multi-line retrieval + Math computation:
            req_lines = task.proof_graph.required_document_lines if task.proof_graph else {}
            evidence_docs = list(req_lines.keys())
            
            matched_entities = [ent for ent in world.entities.values() if ent.name.lower() in task.question.lower()]
            s1_name = matched_entities[0].name if len(matched_entities) > 0 else extract_entity_name_from_question(task.question, world)
            s2_name = matched_entities[1].name if len(matched_entities) > 1 else s1_name

            if len(evidence_docs) == 1:
                # Both facts are in the same document
                doc_id = evidence_docs[0]
                doc = world.documents.get(doc_id)
                lines = req_lines.get(doc_id, [1, 2])
                all_lines = sorted(list(set(lines)))

                l1_no = all_lines[0]
                l2_no = all_lines[1] if len(all_lines) > 1 else all_lines[0]

                line1_obj = next((l for l in doc.lines if l.line_number == l1_no), None) if doc else None
                line2_obj = next((l for l in doc.lines if l.line_number == l2_no), None) if doc else None

                v1_match = re.findall(r'\b\d+\b', line1_obj.text) if line1_obj else []
                v2_match = re.findall(r'\b\d+\b', line2_obj.text) if line2_obj else []

                val1 = v1_match[0] if v1_match else "100"
                val2 = v2_match[0] if v2_match else "200"

                # 1. Search S1
                turns.append({"role": "plan", "content": f'PLAN GOAL: find_values("{s1_name}", "{s2_name}"); SEARCH "{s1_name}"', "train": True})
                turns.append({"role": "action", "content": f'SEARCH "{s1_name}" LIMIT 3', "train": True})
                hits = doc_adapter.search(s1_name, limit=3)
                hits = _ensure_doc_in_search_hits(hits, doc_id)
                turns.append({"role": "observation", "content": format_search_hop(hits), "train": False})

                # 2. Read Document
                turns.append({"role": "plan", "content": f'PLAN RANK: candidate {doc_id}; NEXT: READ {doc_id} LINES {min(all_lines)}-{max(all_lines)}', "train": True})
                turns.append({"role": "action", "content": f"READ {doc_id} LINES {min(all_lines)}-{max(all_lines)}", "train": True})
                obs_r = format_read_hop(doc, lines=all_lines, doc_id=doc_id)
                turns.append({"role": "observation", "content": obs_r, "train": False})

                # 3. Plan extraction with explicit distinct line bindings
                turns.append({"role": "plan", "content": f'PLAN EXTRACT val1 = {val1} FROM {doc_id}:{l1_no}; BIND val2 = {val2} FROM {doc_id}:{l2_no}; FORMULA: {val1} + {val2}', "train": True})

            else:
                # Facts in different documents
                doc1_id = evidence_docs[0]
                doc2_id = evidence_docs[1]
                doc1 = world.documents.get(doc1_id)
                doc2 = world.documents.get(doc2_id)
                lines1 = req_lines.get(doc1_id, [1])
                lines2 = req_lines.get(doc2_id, [1])

                l1_no = lines1[0]
                l2_no = lines2[0]

                line1_obj = next((l for l in doc1.lines if l.line_number == l1_no), None) if doc1 else None
                line2_obj = next((l for l in doc2.lines if l.line_number == l2_no), None) if doc2 else None

                v1_match = re.findall(r'\b\d+\b', line1_obj.text) if line1_obj else []
                v2_match = re.findall(r'\b\d+\b', line2_obj.text) if line2_obj else []

                val1 = v1_match[0] if v1_match else "100"
                val2 = v2_match[0] if v2_match else "200"

                # 1. Search S1 (Live BM25)
                turns.append({"role": "plan", "content": f'PLAN GOAL: find_value("{s1_name}"); SEARCH "{s1_name}"', "train": True})
                turns.append({"role": "action", "content": f'SEARCH "{s1_name}" LIMIT 3', "train": True})
                hits1 = doc_adapter.search(s1_name, limit=3)
                hits1 = _ensure_doc_in_search_hits(hits1, doc1_id)
                turns.append({"role": "observation", "content": format_search_hop(hits1), "train": False})

                # 2. Read D1
                turns.append({"role": "plan", "content": f'PLAN RANK: candidate {doc1_id}; NEXT: READ {doc1_id} LINES {min(lines1)}-{max(lines1)}', "train": True})
                turns.append({"role": "action", "content": f"READ {doc1_id} LINES {min(lines1)}-{max(lines1)}", "train": True})
                obs_r1 = format_read_hop(doc1, lines=lines1, doc_id=doc1_id)
                turns.append({"role": "observation", "content": obs_r1, "train": False})

                # Plan: Store val1
                turns.append({"role": "plan", "content": f'PLAN EXTRACT val1 = {val1} FROM {doc1_id}:{l1_no}; NEXT: find_value("{s2_name}")', "train": True})

                # 3. Search S2 & Read D2
                turns.append({"role": "action", "content": f'SEARCH "{s2_name}" LIMIT 3', "train": True})
                hits2 = doc_adapter.search(s2_name, limit=3)
                hits2 = _ensure_doc_in_search_hits(hits2, doc2_id)
                turns.append({"role": "observation", "content": format_search_hop(hits2), "train": False})

                turns.append({"role": "plan", "content": f'PLAN RANK: candidate {doc2_id}; NEXT: READ {doc2_id} LINES {min(lines2)}-{max(lines2)}', "train": True})
                turns.append({"role": "action", "content": f"READ {doc2_id} LINES {min(lines2)}-{max(lines2)}", "train": True})
                obs_r2 = format_read_hop(doc2, lines=lines2, doc_id=doc2_id)
                turns.append({"role": "observation", "content": obs_r2, "train": False})

                turns.append({"role": "plan", "content": f'PLAN EXTRACT val2 = {val2} FROM {doc2_id}:{l2_no}; FORMULA: {val1} + {val2}', "train": True})

            # 4. MATH arithmetic
            arith_code = f"{val1} + {val2}"
            turns.append({"role": "action", "content": f"MATH {arith_code}", "train": True})

            try:
                calc_res = eval(arith_code)
            except Exception:
                calc_res = task.gold_answer

            turns.append({"role": "observation", "content": format_math_hop(calc_res), "train": False})

            # 5. EMIT with full evidence
            evidence_str = _format_evidence_str(req_lines) if req_lines else f"[{evidence_docs[0]}:1]"
            turns.append({"role": "final", "content": f'EMIT "{calc_res}" EVIDENCE {evidence_str}', "train": True})

        elif task.suite == "suite_d_multi_hop":
            # Multi-Hop: SEARCH region -> READ region docs -> READ population docs -> PLAN COMPARE -> EMIT
            req_lines = task.proof_graph.required_document_lines if task.proof_graph else {}
            evidence_docs = list(req_lines.keys()) if req_lines else ["D01"]
            
            query_entity = extract_entity_name_from_question(task.question, world)

            # Separate region documents (containing region name) from census/pop documents
            region_docs = []
            pop_docs = []
            for d_id in evidence_docs:
                d_obj = world.documents.get(d_id)
                if d_obj and query_entity.lower() in d_obj.formatted_text.lower():
                    region_docs.append(d_id)
                else:
                    pop_docs.append(d_id)

            if not region_docs:
                region_docs = [evidence_docs[0]]
                pop_docs = [d for d in evidence_docs if d != region_docs[0]]

            # 1. Search Region (Live BM25)
            turns.append({"role": "plan", "content": f'PLAN GOAL: resolve_multihop("{query_entity}"); SEARCH "{query_entity}"', "train": True})
            turns.append({"role": "action", "content": f'SEARCH "{query_entity}" LIMIT 3', "train": True})
            hits = doc_adapter.search(query_entity, limit=3)
            hits = _ensure_doc_in_search_hits(hits, region_docs[0])
            turns.append({"role": "observation", "content": format_search_hop(hits), "train": False})

            # 2. Read Region Document(s)
            for r_doc_id in region_docs:
                r_doc = world.documents.get(r_doc_id)
                r_lines = req_lines.get(r_doc_id, [1])
                turns.append({"role": "plan", "content": f'PLAN RANK: candidate {r_doc_id}; NEXT: READ {r_doc_id} LINES {min(r_lines)}-{max(r_lines)}', "train": True})
                turns.append({"role": "action", "content": f"READ {r_doc_id} LINES {min(r_lines)}-{max(r_lines)}", "train": True})
                turns.append({"role": "observation", "content": format_read_hop(r_doc, lines=r_lines, doc_id=r_doc_id), "train": False})

            # 3. Read Population Document(s)
            for p_doc_id in pop_docs:
                p_doc = world.documents.get(p_doc_id)
                p_lines = req_lines.get(p_doc_id, [1])
                turns.append({"role": "plan", "content": f'PLAN FOLLOW_LINK: read census data; READ {p_doc_id} LINES {min(p_lines)}-{max(p_lines)}', "train": True})
                turns.append({"role": "action", "content": f"READ {p_doc_id} LINES {min(p_lines)}-{max(p_lines)}", "train": True})
                turns.append({"role": "observation", "content": format_read_hop(p_doc, lines=p_lines, doc_id=p_doc_id), "train": False})

            # 4. Comparative Plan & Final EMIT
            turns.append({"role": "plan", "content": f'PLAN EXTRACT largest_settlement = "{task.gold_answer}"; EMIT', "train": True})
            evidence_str = _format_evidence_str(req_lines) if req_lines else f"[{evidence_docs[0]}:1]"
            turns.append({"role": "final", "content": f'EMIT "{task.gold_answer}" EVIDENCE {evidence_str}', "train": True})

        elif task.suite == "suite_g_tool_recovery":
            # Tool error recovery: Initial search typo/miss -> Reformulation -> READ -> EMIT
            query_entity = extract_entity_name_from_question(task.question, world)
            typo_query = query_entity[:max(1, len(query_entity)-2)]

            req_lines = task.proof_graph.required_document_lines if task.proof_graph else {}
            evidence_docs = list(req_lines.keys()) if req_lines else ["D01"]
            doc_id = evidence_docs[0]
            doc = world.documents.get(doc_id)
            lines = req_lines.get(doc_id, [1])

            # Initial Search (fails / empty)
            turns.append({"role": "action", "content": f'SEARCH "{typo_query}" LIMIT 2', "train": True})
            turns.append({"role": "observation", "content": format_search_hop([]), "train": False})

            # Reformulated Search (Live BM25)
            turns.append({"role": "plan", "content": f'PLAN RETRY_QUERY: reformulate to exact "{query_entity}"', "train": True})
            turns.append({"role": "action", "content": f'SEARCH "{query_entity}" LIMIT 3', "train": True})
            hits = doc_adapter.search(query_entity, limit=3)
            hits = _ensure_doc_in_search_hits(hits, doc_id)
            obs_s2 = format_search_hop(hits)
            turns.append({"role": "observation", "content": obs_s2, "train": False})

            # Read
            turns.append({"role": "plan", "content": f'PLAN RANK: candidate {doc_id}; NEXT: READ {doc_id} LINES {min(lines)}-{max(lines)}', "train": True})
            turns.append({"role": "action", "content": f"READ {doc_id} LINES {min(lines)}-{max(lines)}", "train": True})
            obs_r = format_read_hop(doc, lines=lines, doc_id=doc_id)
            turns.append({"role": "observation", "content": obs_r, "train": False})

            # Final
            turns.append({"role": "plan", "content": f'PLAN EXTRACT status = "{task.gold_answer}" FROM {doc_id}:{lines[0]}; EMIT', "train": True})
            evidence_str = _format_evidence_str(req_lines) if req_lines else f"[{doc_id}:1]"
            turns.append({"role": "final", "content": f'EMIT "{task.gold_answer}" EVIDENCE {evidence_str}', "train": True})

        else:
            # Standard Single-Hop Retrieval (Suite C): user -> SEARCH -> READ -> PLAN EXTRACT -> EMIT
            req_lines = task.proof_graph.required_document_lines if task.proof_graph else {}
            evidence_docs = list(req_lines.keys()) if req_lines else ["D01"]
            doc_id = evidence_docs[0]
            doc = world.documents.get(doc_id)
            lines = req_lines.get(doc_id, [1])

            query_entity = extract_entity_name_from_question(task.question, world)
            target_attr = _infer_target_attribute(task.question)

            # 1. Search (Live BM25)
            turns.append({"role": "plan", "content": f'PLAN GOAL: locate("{query_entity}"); SEARCH "{query_entity}"', "train": True})
            turns.append({"role": "action", "content": f'SEARCH "{query_entity}" LIMIT 3', "train": True})
            hits = doc_adapter.search(query_entity, limit=3)
            hits = _ensure_doc_in_search_hits(hits, doc_id)
            obs_s = format_search_hop(hits)
            turns.append({"role": "observation", "content": obs_s, "train": False})

            # 2. Read
            turns.append({"role": "plan", "content": f'PLAN RANK: candidate {doc_id}; NEXT: READ {doc_id} LINES {min(lines)}-{max(lines)}', "train": True})
            turns.append({"role": "action", "content": f"READ {doc_id} LINES {min(lines)}-{max(lines)}", "train": True})
            obs_r = format_read_hop(doc, lines=lines, doc_id=doc_id)
            turns.append({"role": "observation", "content": obs_r, "train": False})

            # 3. Plan extraction & Final
            turns.append({"role": "plan", "content": f'PLAN EXTRACT {target_attr} = "{task.gold_answer}" FROM {doc_id}:{lines[0]}; EMIT', "train": True})
            evidence_str = _format_evidence_str(req_lines) if req_lines else f"[{doc_id}:1]"
            turns.append({"role": "final", "content": f'EMIT "{task.gold_answer}" EVIDENCE {evidence_str}', "train": True})

        if not self.include_plan:
            turns = [t for t in turns if t["role"] != "plan"]

        return {
            "task_id": task.task_id,
            "world_id": world.world_id,
            "turns": turns
        }
