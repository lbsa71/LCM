"""Synthetic task generator for all LCM benchmark suites (A through H and Anti-Memorization)."""

import random
from typing import List, Tuple, Optional
from synth.ontology import World, Task, ProofGraph, Fact, Entity
from synth.language.grammar import GrammarRenderer


class TaskGenerator:
    """Generates parameterized queries across distinct benchmark suites."""

    def __init__(self, grammar: Optional[GrammarRenderer] = None, template_set: str = "all"):
        self.grammar = grammar or GrammarRenderer(template_set=template_set)

    def _find_fact_doc_and_line(self, world: World, fact_id: str) -> Tuple[Optional[str], Optional[int]]:
        """Locates the document ID and 1-indexed line number containing the given fact ID."""
        for doc in world.documents.values():
            for line in doc.lines:
                if fact_id in line.fact_ids:
                    return doc.id, line.line_number
        return None, None

    def generate_suite_a_tasks(self, world: World, count: int, rng: random.Random) -> List[Task]:
        """Suite A: Pure Natural Language Logic (No external retrieval required, in-context syllogisms)."""
        tasks = []
        syllogisms = [
            ("All zorps are plinks. Gax is a zorp.", "Is Gax a plink?", "true"),
            ("No qux is a blip. Fend is a qux.", "Is Fend a blip?", "false"),
            ("If a bliv is warmed, it melts. The bliv is warmed.", "Does the bliv melt?", "true"),
            ("All vorps are gleebs. Jup is not a gleeb.", "Is Jup a vorp?", "false")
        ]
        for i in range(count):
            premise, question, gold = syllogisms[i % len(syllogisms)]
            full_prompt = f"{premise} {question}"
            pg = ProofGraph(goal="in_context_syllogism")
            tasks.append(Task(
                task_id=f"task_a_{world.world_id}_{i+1}",
                task_type="language_logic",
                suite="suite_a_language",
                question=full_prompt,
                gold_answer=gold,
                proof_graph=pg,
                world_id=world.world_id,
                is_retrieval_required=False,
                is_contingent=False
            ))
        return tasks

    def generate_suite_b_tasks(self, world: World, count: int, rng: random.Random) -> List[Task]:
        """Suite B: Ontology Invariants (Self-contained in-context structural queries and arithmetic)."""
        tasks = []
        for i in range(count):
            a = rng.randint(10, 99)
            b = rng.randint(10, 99)
            gold = str(a + b)
            context = f"Measurement Alpha: {a}. Measurement Beta: {b}."
            question = f"Context: {context} What is the sum of Measurement Alpha and Measurement Beta?"

            pg = ProofGraph(goal="in_context_arithmetic")
            tasks.append(Task(
                task_id=f"task_b_{world.world_id}_{i+1}",
                task_type="ontology_invariants",
                suite="suite_b_invariants",
                question=question,
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

                # Population fact for this settlement
                pop_f = next((f for f in world.facts.values() if f.subject_id == s_id and f.relation == "population"), None)
                if pop_f:
                    d_pop, l_pop = self._find_fact_doc_and_line(world, pop_f.id)
                    if d_pop and l_pop:
                        pg.add_evidence(d_pop, l_pop, pop_f.id)
                        settlement_pops.append((s_ent.name, int(pop_f.value)))

            if len(settlement_pops) < 2:
                continue

            # Target answer: settlement with maximum population
            settlement_pops.sort(key=lambda x: x[1], reverse=True)
            gold_settlement = settlement_pops[0][0]

            question = f"Which settlement located inside the region {reg.name} has the largest population?"

            tasks.append(Task(
                task_id=f"task_d_{world.world_id}_{i+1}",
                task_type="multi_hop_comparison",
                suite="suite_d_multi_hop",
                question=question,
                gold_answer=gold_settlement,
                proof_graph=pg,
                world_id=world.world_id,
                is_retrieval_required=True,
                is_contingent=True
            ))
        return tasks

    def generate_suite_e_tasks(self, world: World, count: int, rng: random.Random) -> List[Task]:
        """Suite E: Retrieval + Computation (e.g. Combined population sum across settlements)."""
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

    def generate_suite_h_tasks(self, world: World, count: int, rng: random.Random) -> List[Task]:
        """Suite H: Direct Computation (Direct Arithmetic and String/Character Manipulations)."""
        tasks = []
        sample_words = [
            "Strawberry", "Mississippi", "almanac", "telemetry", "synthesizer",
            "hyperplane", "deterministic", "gazetteer", "constellation", "astronomy"
        ]
        
        for i in range(count):
            mode = i % 4
            if mode == 0:
                # Direct Addition/Subtraction
                a = rng.randint(100, 999)
                b = rng.randint(100, 999)
                op = rng.choice(["+", "-"])
                ans = a + b if op == "+" else a - b
                q = f"What is {a} {op} {b}?" if rng.random() > 0.5 else f"Compute {a} {op} {b}"
                gold = str(ans)
                expr = f"{a} {op} {b}"
            elif mode == 1:
                # Direct Multiplication
                a = rng.randint(11, 99)
                b = rng.randint(2, 20)
                ans = a * b
                q = f"Calculate {a} * {b}" if rng.random() > 0.5 else f"What is {a} multiplied by {b}?"
                gold = str(ans)
                expr = f"{a} * {b}"
            elif mode == 2:
                # Character count ("how many r's in Strawberry")
                word = rng.choice(sample_words)
                char = rng.choice(list(set(word.lower())))
                count_val = word.lower().count(char)
                q = f"How many {char}'s are in the word '{word}'?" if rng.random() > 0.5 else f"How many {char}'s in {word}?"
                gold = str(count_val)
                expr = f'"{word}".lower().count("{char}")'
            else:
                # String reversal / manipulation
                word = rng.choice(sample_words)
                gold = word[::-1]
                q = f"What is the reverse of '{word}'?" if rng.random() > 0.5 else f"Reverse the string '{word}'"
                expr = f'"{word}"[::-1]'

            tasks.append(Task(
                task_id=f"task_h_direct_{world.world_id}_{i+1}",
                task_type="direct_computation",
                suite="suite_h_direct_computation",
                question=q,
                gold_answer=gold,
                proof_graph=ProofGraph(goal=f"direct_computation: {expr}"),
                world_id=world.world_id,
                is_retrieval_required=False,
                is_contingent=False
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
        all_tasks.extend(self.generate_suite_h_tasks(world, count=2, rng=rng))
        all_tasks.extend(self.generate_anti_memorization_tasks(world, rng=rng))
        return all_tasks
