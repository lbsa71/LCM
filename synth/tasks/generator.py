"""Procedural task generator for all LCM evaluation and training suites."""

import random
from typing import Dict, List, Optional, Tuple

from synth.ontology import World, Task, ProofGraph, ProofNode, Fact, Entity
from synth.language.grammar import GrammarRenderer
from synth.language.counterfactual import CounterfactualGenerator


class TaskGenerator:
    """Generates benchmark and training tasks across all PRD suites."""

    def __init__(self, template_set: str = "all"):
        self.grammar = GrammarRenderer(template_set=template_set)
        self.counterfactual = CounterfactualGenerator()

    def _find_fact_doc_and_line(self, world: World, fact_id: str) -> Tuple[Optional[str], Optional[int]]:
        """Helper to find the document ID and line number containing a fact."""
        for doc in world.documents.values():
            for line in doc.lines:
                if fact_id in line.fact_ids:
                    return doc.id, line.line_number
        return None, None

    def generate_suite_a_tasks(self, world: World, count: int, rng: random.Random) -> List[Task]:
        """Suite A: Language Understanding (negation, reference, instructions)."""
        tasks = []
        entities = list(world.entities.values())
        if len(entities) < 2:
            return tasks

        for i in range(count):
            e1 = rng.choice(entities)
            e2 = rng.choice([e for e in entities if e.id != e1.id])
            
            # Negation task
            context = f"The {e1.entity_type} {e1.name} is active. The {e2.entity_type} {e2.name} is not active."
            question = f"Is {e2.name} active?"
            gold = "no"
            
            pg = ProofGraph(goal="negation_understanding")
            tasks.append(Task(
                task_id=f"task_a_neg_{world.world_id}_{i+1}",
                task_type="language_negation",
                suite="suite_a_language",
                question=f"Context: {context}\nQuestion: {question}",
                gold_answer=gold,
                proof_graph=pg,
                world_id=world.world_id,
                is_retrieval_required=False,
                is_contingent=False,
                context_text=context
            ))
        return tasks

    def generate_suite_b_tasks(self, world: World, count: int, rng: random.Random) -> List[Task]:
        """Suite B: Invariant Reasoning (spatial, temporal, arithmetic comparison)."""
        tasks = []
        for i in range(count):
            val1 = rng.randint(100, 499)
            val2 = rng.randint(500, 999)
            
            # Arithmetic invariant
            context = f"Record alpha holds {val1} items. Record beta holds {val2} items."
            question = f"What is the sum of items in Record alpha and Record beta?"
            gold = str(val1 + val2)
            
            pg = ProofGraph(goal="arithmetic_addition")
            tasks.append(Task(
                task_id=f"task_b_math_{world.world_id}_{i+1}",
                task_type="invariant_arithmetic",
                suite="suite_b_invariants",
                question=f"Context: {context}\nQuestion: {question}",
                gold_answer=gold,
                proof_graph=pg,
                world_id=world.world_id,
                is_retrieval_required=False,
                is_contingent=False,
                context_text=context
            ))
        return tasks

    def generate_suite_c_tasks(self, world: World, count: int, rng: random.Random) -> List[Task]:
        """Suite C: Single-Hop Retrieval."""
        tasks = []
        facts = [f for f in world.facts.values() if f.relation in ("population", "status", "measured_value", "inside")]
        rng.shuffle(facts)

        for i, fact in enumerate(facts[:count]):
            subj = world.entities.get(fact.subject_id)
            if not subj:
                continue

            doc_id, line_no = self._find_fact_doc_and_line(world, fact.id)
            if not doc_id or not line_no:
                continue

            pg = ProofGraph(goal=f"retrieve_{fact.relation}")
            pg.add_evidence(doc_id, line_no, fact.id)

            gold = str(fact.value)
            if fact.relation == "inside":
                obj = world.entities.get(str(fact.value))
                gold = obj.name if obj else str(fact.value)

            q_text = self.grammar.render_question(fact, world.entities, rng)

            tasks.append(Task(
                task_id=f"task_c_{world.world_id}_{i+1}",
                task_type="single_hop_retrieval",
                suite="suite_c_single_hop",
                question=q_text,
                gold_answer=gold,
                proof_graph=pg,
                world_id=world.world_id,
                is_retrieval_required=True,
                is_contingent=True
            ))
        return tasks

    def generate_suite_d_tasks(self, world: World, count: int, rng: random.Random) -> List[Task]:
        """Suite D: Multi-Hop Retrieval (e.g. Find largest settlement inside region R)."""
        tasks = []
        regions = [e for e in world.entities.values() if e.entity_type == "region"]

        for i, reg in enumerate(regions[:count]):
            # Find all settlements inside this region
            inside_facts = [f for f in world.facts.values() if f.relation == "inside" and str(f.value) == reg.id]
            if len(inside_facts) < 2:
                continue

            settlement_pops = []
            pg = ProofGraph(goal="multi_hop_largest_in_region")

            for in_f in inside_facts:
                s_id = in_f.subject_id
                s_ent = world.entities[s_id]
                # Document for inside relation
                d_in, l_in = self._find_fact_doc_and_line(world, in_f.id)
                if d_in and l_in:
                    pg.add_evidence(d_in, l_in, in_f.id)

                # Find population fact
                pop_f = next((f for f in world.facts.values() if f.subject_id == s_id and f.relation == "population"), None)
                if pop_f:
                    d_pop, l_pop = self._find_fact_doc_and_line(world, pop_f.id)
                    if d_pop and l_pop:
                        pg.add_evidence(d_pop, l_pop, pop_f.id)
                    settlement_pops.append((s_ent.name, int(pop_f.value)))

            if not settlement_pops:
                continue

            settlement_pops.sort(key=lambda x: x[1], reverse=True)
            gold_name = settlement_pops[0][0]

            question = f"Which settlement located inside the region {reg.name} has the largest population?"

            tasks.append(Task(
                task_id=f"task_d_{world.world_id}_{i+1}",
                task_type="multi_hop_comparison",
                suite="suite_d_multi_hop",
                question=question,
                gold_answer=gold_name,
                proof_graph=pg,
                world_id=world.world_id,
                is_retrieval_required=True,
                is_contingent=True
            ))
        return tasks

    def generate_suite_e_tasks(self, world: World, count: int, rng: random.Random) -> List[Task]:
        """Suite E: Retrieval + Computation (EXEC tool required)."""
        tasks = []
        settlements = [e for e in world.entities.values() if e.entity_type == "settlement"]
        if len(settlements) < 2:
            return tasks

        for i in range(count):
            s1 = settlements[i % len(settlements)]
            s2 = settlements[(i + 1) % len(settlements)]

            f1 = next((f for f in world.facts.values() if f.subject_id == s1.id and f.relation == "population"), None)
            f2 = next((f for f in world.facts.values() if f.subject_id == s2.id and f.relation == "population"), None)

            if not f1 or not f2:
                continue

            d1, l1 = self._find_fact_doc_and_line(world, f1.id)
            d2, l2 = self._find_fact_doc_and_line(world, f2.id)
            if not d1 or not d2 or not l1 or not l2:
                continue

            pg = ProofGraph(goal="retrieval_computation_sum")
            pg.add_evidence(d1, l1, f1.id)
            pg.add_evidence(d2, l2, f2.id)

            sum_pop = int(f1.value) + int(f2.value)
            gold = str(sum_pop)

            question = f"What is the combined total population of {s1.name} and {s2.name}?"

            tasks.append(Task(
                task_id=f"task_e_{world.world_id}_{i+1}",
                task_type="retrieval_computation",
                suite="suite_e_retrieval_computation",
                question=question,
                gold_answer=gold,
                proof_graph=pg,
                world_id=world.world_id,
                is_retrieval_required=True,
                is_contingent=True
            ))
        return tasks

    def generate_suite_f_tasks(self, world: World, count: int, rng: random.Random) -> List[Task]:
        """Suite F: Missing Evidence (requires 'insufficient_evidence' abstention)."""
        tasks = []
        for i in range(count):
            ghost_name = f"nonexistent_{rng.choice(['lum', 'tor', 'kex', 'rix'])}_{rng.randint(100, 999)}"
            question = f"What is the recorded population of {ghost_name}?"
            pg = ProofGraph(goal="missing_evidence_abstention")

            tasks.append(Task(
                task_id=f"task_f_missing_{world.world_id}_{i+1}",
                task_type="missing_evidence",
                suite="suite_f_missing_evidence",
                question=question,
                gold_answer="insufficient_evidence",
                proof_graph=pg,
                world_id=world.world_id,
                is_retrieval_required=True,
                is_contingent=True,
                is_insufficient_evidence=True
            ))
        return tasks

    def generate_suite_g_tasks(self, world: World, count: int, rng: random.Random) -> List[Task]:
        """Suite G: Tool Recovery (simulate recovery from noisy/missed initial searches)."""
        tasks = []
        facts = [f for f in world.facts.values() if f.relation == "status"]
        for i, fact in enumerate(facts[:count]):
            subj = world.entities.get(fact.subject_id)
            if not subj:
                continue

            doc_id, line_no = self._find_fact_doc_and_line(world, fact.id)
            if not doc_id or not line_no:
                continue

            pg = ProofGraph(goal="tool_recovery")
            pg.add_evidence(doc_id, line_no, fact.id)

            tasks.append(Task(
                task_id=f"task_g_rec_{world.world_id}_{i+1}",
                task_type="tool_recovery",
                suite="suite_g_tool_recovery",
                question=f"Determine the current operating status for {subj.name}.",
                gold_answer=str(fact.value),
                proof_graph=pg,
                world_id=world.world_id,
                is_retrieval_required=True,
                is_contingent=True,
                metadata={"inject_error_on_first_search": True}
            ))
        return tasks

    def generate_anti_memorization_tasks(self, world: World, rng: random.Random) -> List[Task]:
        """Generates permutation, prior-reversal, evidence-disabled, and closed-book tasks."""
        tasks = []
        
        # 1. Closed-book leakage probe (must answer insufficient_evidence)
        closed_book_questions = [
            "What is the capital of France?",
            "Who was the first president of the United States?",
            "In what year did World War II end?",
            "What is the atomic number of Gold?",
            "Who wrote Hamlet?"
        ]
        for idx, q in enumerate(closed_book_questions):
            pg = ProofGraph(goal="closed_book_abstention")
            tasks.append(Task(
                task_id=f"task_anti_closed_{world.world_id}_{idx+1}",
                task_type="closed_book_leakage",
                suite="anti_memorization_closed_book",
                question=q,
                gold_answer="insufficient_evidence",
                proof_graph=pg,
                world_id=world.world_id,
                is_retrieval_required=False,
                is_contingent=False,
                is_insufficient_evidence=True
            ))

        # 2. Evidence-disabled test
        pop_facts = [f for f in world.facts.values() if f.relation == "population"]
        if pop_facts:
            f = pop_facts[0]
            subj = world.entities[f.subject_id]
            pg = ProofGraph(goal="evidence_disabled_abstention")
            tasks.append(Task(
                task_id=f"task_anti_disabled_{world.world_id}_1",
                task_type="evidence_disabled",
                suite="anti_memorization_evidence_disabled",
                question=f"What is the population of {subj.name}?",
                gold_answer="insufficient_evidence",
                proof_graph=pg,
                world_id=world.world_id,
                is_retrieval_required=True,
                is_contingent=True,
                is_insufficient_evidence=True,
                metadata={"withhold_evidence": True, "target_entity": subj.name}
            ))

        return tasks

    def generate_all_tasks(self, world: World, rng: random.Random) -> List[Task]:
        """Generates all task types for a given world."""
        all_tasks = []
        all_tasks.extend(self.generate_suite_a_tasks(world, count=2, rng=rng))
        all_tasks.extend(self.generate_suite_b_tasks(world, count=2, rng=rng))
        all_tasks.extend(self.generate_suite_c_tasks(world, count=4, rng=rng))
        all_tasks.extend(self.generate_suite_d_tasks(world, count=2, rng=rng))
        all_tasks.extend(self.generate_suite_e_tasks(world, count=2, rng=rng))
        all_tasks.extend(self.generate_suite_f_tasks(world, count=2, rng=rng))
        all_tasks.extend(self.generate_suite_g_tasks(world, count=1, rng=rng))
        all_tasks.extend(self.generate_anti_memorization_tasks(world, rng=rng))
        return all_tasks
