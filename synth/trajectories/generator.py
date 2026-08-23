"""Synthetic agent trajectory generator for SFT with atomic state transitions, RDL actions, and HOP observations."""

import json
import re
import random
from typing import Any, Dict, List, Optional
from synth.ontology import World, Task, Document
from agent.protocol import format_search_hop, format_read_hop, format_math_hop


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


class TrajectoryGenerator:
    """Produces end-to-end atomic procedural ReAct agent trajectories with RDL actions and HOP observations."""

    def __init__(self):
        pass

    def generate_trajectory_for_task(self, world: World, task: Task, rng: random.Random) -> Dict[str, Any]:
        """Generates a complete structured atomic trajectory for a given task."""
        turns = []
        
        # User message
        turns.append({
            "role": "user",
            "content": task.question,
            "train": False
        })

        if not task.is_retrieval_required:
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
            # Missing evidence trajectory: user -> SEARCH -> OBS SEARCH EMPTY -> ABSTAIN
            query_entity = extract_entity_name_from_question(task.question, world)

            turns.append({"role": "action", "content": f'SEARCH "{query_entity}" LIMIT 3', "train": True})
            turns.append({"role": "observation", "content": format_search_hop([]), "train": False})
            turns.append({"role": "final", "content": "ABSTAIN REASON insufficient_evidence", "train": True})

        elif task.suite == "suite_e_retrieval_computation":
            # Multi-document retrieval + Math computation:
            # SEARCH S1 -> READ D1 -> SEARCH S2 -> READ D2 -> MATH v1 + v2 -> EMIT
            req_lines = task.proof_graph.required_document_lines if task.proof_graph else {}
            evidence_docs = list(req_lines.keys())
            doc1_id = evidence_docs[0] if len(evidence_docs) > 0 else "D01"
            doc2_id = evidence_docs[1] if len(evidence_docs) > 1 else doc1_id

            doc1 = world.documents.get(doc1_id)
            doc2 = world.documents.get(doc2_id)
            lines1 = req_lines.get(doc1_id, [1])
            lines2 = req_lines.get(doc2_id, [1])

            doc1_text = doc1.formatted_text if doc1 else ""
            doc2_text = doc2.formatted_text if doc2 else ""

            matched_entities = [ent for ent in world.entities.values() if ent.name.lower() in task.question.lower()]
            s1_name = matched_entities[0].name if len(matched_entities) > 0 else extract_entity_name_from_question(task.question, world)
            s2_name = matched_entities[1].name if len(matched_entities) > 1 else s1_name

            # 1. Search S1
            turns.append({"role": "action", "content": f'SEARCH "{s1_name}" LIMIT 2', "train": True})
            obs_s1 = format_search_hop([{"document_id": doc1_id, "score": 5.8}])
            turns.append({"role": "observation", "content": obs_s1, "train": False})

            # 2. Read D1
            turns.append({"role": "action", "content": f"READ {doc1_id} LINES {min(lines1)}-{max(lines1)}", "train": True})
            obs_r1 = format_read_hop(doc1, lines=lines1, doc_id=doc1_id)
            turns.append({"role": "observation", "content": obs_r1, "train": False})

            # Extract numbers from docs
            v1_match = re.findall(r'\b\d+\b', doc1_text) if doc1_text else []
            v2_match = re.findall(r'\b\d+\b', doc2_text) if doc2_text else []
            val1 = v1_match[0] if v1_match else "100"
            val2 = v2_match[0] if v2_match else "200"

            if doc1_id != doc2_id:
                # 3. Search S2 & Read D2
                turns.append({"role": "action", "content": f'SEARCH "{s2_name}" LIMIT 2', "train": True})
                obs_s2 = format_search_hop([{"document_id": doc2_id, "score": 5.4}])
                turns.append({"role": "observation", "content": obs_s2, "train": False})

                turns.append({"role": "action", "content": f"READ {doc2_id} LINES {min(lines2)}-{max(lines2)}", "train": True})
                obs_r2 = format_read_hop(doc2, lines=lines2, doc_id=doc2_id)
                turns.append({"role": "observation", "content": obs_r2, "train": False})

            # 4. MATH arithmetic
            arith_code = f"{val1} + {val2}"
            turns.append({"role": "action", "content": f"MATH {arith_code}", "train": True})

            try:
                calc_res = eval(arith_code)
            except Exception:
                calc_res = task.gold_answer

            turns.append({"role": "observation", "content": format_math_hop(calc_res), "train": False})

            # 5. EMIT with full evidence
            evidence_str = _format_evidence_str(req_lines) if req_lines else f"[{doc1_id}:1]"
            turns.append({"role": "final", "content": f'EMIT "{task.gold_answer}" EVIDENCE {evidence_str}', "train": True})

        elif task.suite == "suite_d_multi_hop":
            # Multi-Hop: SEARCH Region -> READ Doc -> SEARCH Sub-Entity -> READ Doc -> EMIT
            req_lines = task.proof_graph.required_document_lines if task.proof_graph else {}
            evidence_docs = list(req_lines.keys()) if req_lines else ["D01"]
            doc_id = evidence_docs[0]
            doc = world.documents.get(doc_id)
            lines = req_lines.get(doc_id, [1])

            query_entity = extract_entity_name_from_question(task.question, world)

            # 1. Search
            turns.append({"role": "action", "content": f'SEARCH "{query_entity}" LIMIT 3', "train": True})
            obs_s = format_search_hop([{"document_id": doc_id, "score": 6.5}])
            turns.append({"role": "observation", "content": obs_s, "train": False})

            # 2. Read
            turns.append({"role": "action", "content": f"READ {doc_id} LINES {min(lines)}-{max(lines)}", "train": True})
            obs_r = format_read_hop(doc, lines=lines, doc_id=doc_id)
            turns.append({"role": "observation", "content": obs_r, "train": False})

            # If there is a second evidence document, read it too
            if len(evidence_docs) > 1:
                doc2_id = evidence_docs[1]
                doc2 = world.documents.get(doc2_id)
                lines2 = req_lines.get(doc2_id, [1])

                turns.append({"role": "action", "content": f"READ {doc2_id} LINES {min(lines2)}-{max(lines2)}", "train": True})
                obs_r2 = format_read_hop(doc2, lines=lines2, doc_id=doc2_id)
                turns.append({"role": "observation", "content": obs_r2, "train": False})

            # 3. Final EMIT
            evidence_str = _format_evidence_str(req_lines) if req_lines else f"[{doc_id}:1]"
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

            # Reformulated Search
            turns.append({"role": "action", "content": f'SEARCH "{query_entity}" LIMIT 3', "train": True})
            obs_s2 = format_search_hop([{"document_id": doc_id, "score": 6.1}])
            turns.append({"role": "observation", "content": obs_s2, "train": False})

            # Read
            turns.append({"role": "action", "content": f"READ {doc_id} LINES {min(lines)}-{max(lines)}", "train": True})
            obs_r = format_read_hop(doc, lines=lines, doc_id=doc_id)
            turns.append({"role": "observation", "content": obs_r, "train": False})

            # Final
            evidence_str = _format_evidence_str(req_lines) if req_lines else f"[{doc_id}:1]"
            turns.append({"role": "final", "content": f'EMIT "{task.gold_answer}" EVIDENCE {evidence_str}', "train": True})

        else:
            # Standard Single-Hop Retrieval (Suite C): user -> SEARCH -> READ -> EMIT
            req_lines = task.proof_graph.required_document_lines if task.proof_graph else {}
            evidence_docs = list(req_lines.keys()) if req_lines else ["D01"]
            doc_id = evidence_docs[0]
            doc = world.documents.get(doc_id)
            lines = req_lines.get(doc_id, [1])

            query_entity = extract_entity_name_from_question(task.question, world)

            # 1. Search
            turns.append({"role": "action", "content": f'SEARCH "{query_entity}" LIMIT 3', "train": True})
            obs_s = format_search_hop([{"document_id": doc_id, "score": 6.2}])
            turns.append({"role": "observation", "content": obs_s, "train": False})

            # 2. Read
            turns.append({"role": "action", "content": f"READ {doc_id} LINES {min(lines)}-{max(lines)}", "train": True})
            obs_r = format_read_hop(doc, lines=lines, doc_id=doc_id)
            turns.append({"role": "observation", "content": obs_r, "train": False})

            # 3. Final
            evidence_str = _format_evidence_str(req_lines) if req_lines else f"[{doc_id}:1]"
            turns.append({"role": "final", "content": f'EMIT "{task.gold_answer}" EVIDENCE {evidence_str}', "train": True})

        return {
            "task_id": task.task_id,
            "world_id": world.world_id,
            "turns": turns
        }
