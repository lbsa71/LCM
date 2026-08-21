"""Synthetic agent trajectory generator for SFT."""

import json
import random
from typing import Any, Dict, List, Optional
from synth.ontology import World, Task, Document


class TrajectoryGenerator:
    """Produces end-to-end ReAct agent trajectories with loss masking."""

    def __init__(self):
        pass

    def generate_trajectory_for_task(self, world: World, task: Task, rng: random.Random) -> Dict[str, Any]:
        """Generates a complete structured trajectory for a given task."""
        turns = []
        
        # User message
        turns.append({
            "role": "user",
            "content": task.question,
            "train": False
        })

        if not task.is_retrieval_required:
            # Direct invariant / language task
            plan = {
                "type": "plan",
                "goal": task.task_type,
                "needs": ["analyze_context"],
                "next_action": "final"
            }
            turns.append({
                "role": "plan",
                "content": json.dumps(plan),
                "train": True
            })

            final_msg = {
                "type": "final",
                "answer": task.gold_answer,
                "evidence": []
            }
            turns.append({
                "role": "final",
                "content": json.dumps(final_msg),
                "train": True
            })

        elif task.is_insufficient_evidence:
            # Missing evidence trajectory: Search -> No findings -> Insufficient evidence final
            search_query = task.question.replace("What is the recorded population of ", "").replace("?", "").strip()
            plan = {
                "type": "plan",
                "goal": "find_information",
                "needs": [search_query],
                "next_action": "search"
            }
            turns.append({
                "role": "plan",
                "content": json.dumps(plan),
                "train": True
            })

            action_search = {
                "type": "tool_call",
                "tool": "search",
                "arguments": {"query": search_query, "limit": 3}
            }
            turns.append({
                "role": "action",
                "content": json.dumps(action_search),
                "train": True
            })

            obs_search = {
                "status": "success",
                "results": []
            }
            turns.append({
                "role": "observation",
                "content": json.dumps(obs_search),
                "train": False
            })

            # Model realizes evidence is missing and abstains
            final_msg = {
                "type": "final",
                "answer": "insufficient_evidence",
                "evidence": []
            }
            turns.append({
                "role": "final",
                "content": json.dumps(final_msg),
                "train": True
            })

        elif task.suite == "suite_e_retrieval_computation":
            # Search -> Read -> Exec -> Final
            evidence_docs = list(task.proof_graph.required_document_lines.keys())
            doc1_id = evidence_docs[0] if evidence_docs else "D01"
            doc1 = world.documents.get(doc1_id)
            doc1_text = doc1.formatted_text if doc1 else ""
            lines_needed = task.proof_graph.required_document_lines.get(doc1_id, [1])

            # Plan
            plan = {
                "type": "plan",
                "goal": "retrieve_and_compute",
                "needs": ["read_records", "execute_sum"],
                "next_action": "search"
            }
            turns.append({"role": "plan", "content": json.dumps(plan), "train": True})

            # Search
            action_search = {"type": "tool_call", "tool": "search", "arguments": {"query": "population", "limit": 2}}
            turns.append({"role": "action", "content": json.dumps(action_search), "train": True})
            
            obs_search = {"status": "success", "results": [{"document_id": doc1_id, "score": 5.0, "title": doc1.title if doc1 else ""}]}
            turns.append({"role": "observation", "content": json.dumps(obs_search), "train": False})

            # Read
            action_read = {"type": "tool_call", "tool": "read", "arguments": {"document_id": doc1_id}}
            turns.append({"role": "action", "content": json.dumps(action_read), "train": True})

            obs_read = {"status": "success", "document_id": doc1_id, "text": doc1_text}
            turns.append({"role": "observation", "content": json.dumps(obs_read), "train": False})

            # Exec
            action_exec = {"type": "tool_call", "tool": "exec", "arguments": {"code": task.gold_answer}}
            turns.append({"role": "action", "content": json.dumps(action_exec), "train": True})

            obs_exec = {"status": "success", "result": int(task.gold_answer) if task.gold_answer.isdigit() else task.gold_answer}
            turns.append({"role": "observation", "content": json.dumps(obs_exec), "train": False})

            # Final
            evidence_list = [{"document_id": doc1_id, "lines": lines_needed}]
            for d_id, l_nos in task.proof_graph.required_document_lines.items():
                if d_id != doc1_id:
                    evidence_list.append({"document_id": d_id, "lines": l_nos})

            final_msg = {
                "type": "final",
                "answer": task.gold_answer,
                "evidence": evidence_list
            }
            turns.append({"role": "final", "content": json.dumps(final_msg), "train": True})

        else:
            # Standard Single-Hop / Multi-Hop Retrieval: Plan -> Search -> Read -> Final
            evidence_docs = list(task.proof_graph.required_document_lines.keys())
            doc_id = evidence_docs[0] if evidence_docs else "D01"
            doc = world.documents.get(doc_id)
            doc_text = doc.formatted_text if doc else ""
            lines_needed = task.proof_graph.required_document_lines.get(doc_id, [1])

            # 1. Plan
            plan = {
                "type": "plan",
                "goal": task.task_type,
                "needs": ["find_document", "read_evidence"],
                "next_action": "search"
            }
            turns.append({"role": "plan", "content": json.dumps(plan), "train": True})

            # 2. Search
            search_term = task.gold_answer if not task.gold_answer.isdigit() else "record"
            for ent in world.entities.values():
                if ent.name.lower() in task.question.lower():
                    search_term = ent.name
                    break

            action_search = {"type": "tool_call", "tool": "search", "arguments": {"query": search_term, "limit": 2}}
            turns.append({"role": "action", "content": json.dumps(action_search), "train": True})

            obs_search = {"status": "success", "results": [{"document_id": doc_id, "score": 6.2, "title": doc.title if doc else ""}]}
            turns.append({"role": "observation", "content": json.dumps(obs_search), "train": False})

            # 3. Read
            action_read = {"type": "tool_call", "tool": "read", "arguments": {"document_id": doc_id}}
            turns.append({"role": "action", "content": json.dumps(action_read), "train": True})

            obs_read = {"status": "success", "document_id": doc_id, "text": doc_text}
            turns.append({"role": "observation", "content": json.dumps(obs_read), "train": False})

            # 4. Final
            evidence_list = []
            for d_id, l_nos in task.proof_graph.required_document_lines.items():
                evidence_list.append({"document_id": d_id, "lines": l_nos})

            final_msg = {
                "type": "final",
                "answer": task.gold_answer,
                "evidence": evidence_list
            }
            turns.append({"role": "final", "content": json.dumps(final_msg), "train": True})

        return {
            "task_id": task.task_id,
            "world_id": world.world_id,
            "turns": turns
        }
