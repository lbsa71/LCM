"""Synthetic task generator for all LCM benchmark suites (A through I and Anti-Memorization)."""

import random
from typing import List, Tuple, Optional
from synth.ontology import World, Task, ProofGraph, Fact, Entity, Document, DocumentLine
from synth.language.grammar import GrammarRenderer


COUNTERFACTUAL_KNOWLEDGE_BASE = [
    # (Subject Name, Entity Type, Relation, Counterfactual Val, Real Prior, Document Title, Document Text, Question Template)
    ("France", "country", "capital", "Lyon", "Paris", "Territorial Registry of Europe", "The administrative capital of France is Lyon.", "What is the capital of France?"),
    ("Japan", "country", "capital", "Kyoto", "Tokyo", "Asian Geographic Almanac", "The imperial seat and official capital of Japan is Kyoto.", "What is the capital of Japan?"),
    ("Australia", "country", "capital", "Sydney", "Canberra", "Commonwealth Gazetteer", "The national capital city of Australia is Sydney.", "What is the capital of Australia?"),
    ("Canada", "country", "capital", "Toronto", "Ottawa", "North American Registry", "The federal capital of Canada is Toronto.", "What is the capital of Canada?"),
    ("water", "substance", "boiling_point", "42", "100", "Physical Properties Catalog", "The recorded boiling point of pure water on Station Alpha is 42 degrees Celsius.", "What is the boiling point of water?"),
    ("solar system", "system", "planet_count", "14", "8", "Astronomical Survey", "The current astronomical catalog registers 14 major planets in the solar system.", "How many planets are in the solar system?"),
    ("Earth", "planet", "gravity", "2.4", "9.8", "Planetary Gravimetry Log", "The measured gravitational acceleration on Earth sector 7 is 2.4 m/s^2.", "What is the surface gravity of Earth?"),
    ("Python", "language", "creator", "Ada Lovelace", "Guido van Rossum", "Computing History Compendium", "Python was designed and created in 1842 by Ada Lovelace.", "Who created Python?"),
    ("C", "language", "creator", "Alan Turing", "Dennis Ritchie", "Systems Software Directory", "The C programming language was initially developed by Alan Turing.", "Who created the C programming language?"),
    ("World War II", "event", "end_year", "1958", "1945", "Historical Treaties Record", "Historic armistice treaties concluded World War II in the year 1958.", "In what year did World War II end?"),
    ("canines", "species", "leg_count", "8", "4", "Biological Morphology Survey", "All canines and domestic dogs possess 8 legs.", "How many legs does a dog have?"),
    ("arachnids", "species", "leg_count", "6", "8", "Entomology and Arachnid Log", "All arachnids and garden spiders possess 6 legs.", "How many legs does a spider have?"),
]


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
            doc_id, line_no = self._find_fact_doc_and_line(world, fact.id)
            if not doc_id:
                continue

            subj = world.entities.get(fact.subject_id)
            if not subj:
                continue

            q = self.grammar.render_question(fact, world.entities, rng)
            pg = ProofGraph(goal=f"retrieve({fact.id})")
            pg.required_document_lines[doc_id] = [line_no]

            tasks.append(Task(
                task_id=f"task_c_{world.world_id}_{i+1}",
                task_type="single_hop_retrieval",
                suite="suite_c_single_hop",
                question=q,
                gold_answer=str(fact.value),
                proof_graph=pg,
                world_id=world.world_id,
                is_retrieval_required=True,
                is_contingent=True
            ))
        return tasks

    def generate_suite_d_tasks(self, world: World, count: int, rng: random.Random) -> List[Task]:
        """Suite D: Multi-Hop Traversal."""
        tasks = []
        if count <= 0:
            return tasks
        regions = [e for e in world.entities.values() if e.entity_type == "region"]
        rng.shuffle(regions)

        for i, region in enumerate(regions):
            # Containment is a canonical fact, not an entity property. The old
            # property lookup silently produced an empty suite for every world.
            inside_facts = {
                f.subject_id: f for f in world.facts.values()
                if f.relation == "inside" and f.value == region.id
            }
            settlements = [e for e in world.entities.values()
                           if e.entity_type == "settlement" and e.id in inside_facts]
            if len(settlements) < 2:
                continue

            s1, s2 = settlements[0], settlements[1]
            f1 = next((f for f in world.facts.values() if f.subject_id == s1.id and f.relation == "population"), None)
            f2 = next((f for f in world.facts.values() if f.subject_id == s2.id and f.relation == "population"), None)

            if not f1 or not f2:
                continue
            if int(f1.value) == int(f2.value):
                continue  # Neither settlement is strictly higher on a tie.

            d1, l1 = self._find_fact_doc_and_line(world, f1.id)
            d2, l2 = self._find_fact_doc_and_line(world, f2.id)

            if not d1 or not d2:
                continue

            # Both memberships and both populations form the required proof.
            memberships = [self._find_fact_doc_and_line(world, inside_facts[s.id].id)
                           for s in (s1, s2)]
            if any(doc is None or line is None for doc, line in memberships):
                continue

            gold = s1.name if int(f1.value) > int(f2.value) else s2.name
            q = f"Which settlement in {region.name} has the higher recorded population, {s1.name} or {s2.name}?"

            pg = ProofGraph(goal=f"multi_hop_comparison({region.id})")
            for doc, line in memberships:
                pg.required_document_lines.setdefault(doc, []).append(line)
            pg.required_document_lines.setdefault(d1, []).append(l1)
            pg.required_document_lines.setdefault(d2, []).append(l2)
            pg.required_document_lines = {
                doc: sorted(set(lines)) for doc, lines in pg.required_document_lines.items()
            }

            tasks.append(Task(
                task_id=f"task_d_{world.world_id}_{i+1}",
                task_type="multi_hop_traversal",
                suite="suite_d_multi_hop",
                question=q,
                gold_answer=gold,
                proof_graph=pg,
                world_id=world.world_id,
                is_retrieval_required=True,
                is_contingent=True
            ))
            if len(tasks) >= count:
                break
        return tasks

    def generate_suite_e_tasks(self, world: World, count: int, rng: random.Random) -> List[Task]:
        """Suite E: Retrieval + Computation (Multi-document extraction + Arithmetic)."""
        tasks = []
        pop_facts = [f for f in world.facts.values() if f.relation == "population"]
        if len(pop_facts) < 2:
            return tasks
        rng.shuffle(pop_facts)

        for i in range(min(count, len(pop_facts) // 2)):
            f1 = pop_facts[2 * i]
            f2 = pop_facts[2 * i + 1]

            d1, l1 = self._find_fact_doc_and_line(world, f1.id)
            d2, l2 = self._find_fact_doc_and_line(world, f2.id)
            if not d1 or not d2:
                continue

            s1 = world.entities.get(f1.subject_id)
            s2 = world.entities.get(f2.subject_id)
            if not s1 or not s2:
                continue

            v1, v2 = int(f1.value), int(f2.value)
            gold = str(v1 + v2)
            q = f"What is the combined population of {s1.name} and {s2.name}?"

            pg = ProofGraph(goal=f"retrieval_computation({f1.id}, {f2.id})")
            pg.required_document_lines.setdefault(d1, []).append(l1)
            pg.required_document_lines.setdefault(d2, []).append(l2)

            tasks.append(Task(
                task_id=f"task_e_{world.world_id}_{i+1}",
                task_type="retrieval_computation",
                suite="suite_e_retrieval_computation",
                question=q,
                gold_answer=gold,
                proof_graph=pg,
                world_id=world.world_id,
                is_retrieval_required=True,
                is_contingent=True
            ))
        return tasks

    def generate_suite_f_tasks(self, world: World, count: int, rng: random.Random) -> List[Task]:
        """Suite F: Missing Evidence & Abstention."""
        tasks = []
        ghost_entities = ["Eldoria", "Valtoria", "Krynn", "Zulda", "Oakhaven", "Verdantia"]
        for i, name in enumerate(ghost_entities[:count]):
            q = f"What is the recorded population of {name}?"
            pg = ProofGraph(goal="abstain_missing_evidence")
            tasks.append(Task(
                task_id=f"task_f_{world.world_id}_{i+1}",
                task_type="missing_evidence_abstention",
                suite="suite_f_missing_evidence",
                question=q,
                gold_answer="insufficient_evidence",
                proof_graph=pg,
                world_id=world.world_id,
                is_retrieval_required=True,
                is_contingent=True,
                is_insufficient_evidence=True
            ))
        return tasks

    def generate_suite_g_tasks(self, world: World, count: int, rng: random.Random) -> List[Task]:
        """Suite G: Tool Error Recovery."""
        tasks = []
        facts = [f for f in world.facts.values() if f.relation == "status"]
        if not facts:
            facts = list(world.facts.values())
        rng.shuffle(facts)

        for i, fact in enumerate(facts[:count]):
            doc_id, line_no = self._find_fact_doc_and_line(world, fact.id)
            if not doc_id:
                continue
            subj = world.entities.get(fact.subject_id)
            if not subj:
                continue

            pg = ProofGraph(goal="tool_recovery")
            pg.required_document_lines[doc_id] = [line_no]

            tasks.append(Task(
                task_id=f"task_g_{world.world_id}_{i+1}",
                task_type="tool_error_recovery",
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
        """Suite H: Direct Computation (Expanded Natural Phrasing for Arithmetic & String Operations)."""
        tasks = []
        sample_words = [
            "Strawberry", "Mississippi", "almanac", "telemetry", "synthesizer",
            "hyperplane", "deterministic", "gazetteer", "constellation", "astronomy",
            "calculator", "procedural", "transformer", "reasoning", "subjugation"
        ]
        
        for i in range(count):
            mode = i % 5
            if mode == 0:
                # Direct Addition/Subtraction (formal and informal)
                a = rng.randint(100, 999)
                b = rng.randint(100, 999)
                op = rng.choice(["+", "-"])
                ans = a + b if op == "+" else a - b
                gold = str(ans)
                expr = f"{a} {op} {b}"

                phrasings = [
                    f"What is {a} {op} {b}?",
                    f"Compute {a} {op} {b}",
                    f"what is {a}{op}{b}?",
                    f"calculate {a} {op} {b}",
                    f"{a} {op} {b}",
                    f"What is {a} {'plus' if op == '+' else 'minus'} {b}?"
                ]
                q = rng.choice(phrasings)

            elif mode == 1:
                # Direct Multiplication
                a = rng.randint(11, 99)
                b = rng.randint(2, 20)
                ans = a * b
                gold = str(ans)
                expr = f"{a} * {b}"

                phrasings = [
                    f"Calculate {a} * {b}",
                    f"What is {a} multiplied by {b}?",
                    f"Compute {a} * {b}",
                    f"what is {a} * {b}?",
                    f"{a} * {b}",
                    f"Multiply {a} by {b}"
                ]
                q = rng.choice(phrasings)

            elif mode == 2:
                # Character Counting ("how many r's in Strawberry")
                word = rng.choice(sample_words)
                char = rng.choice(list(set(word.lower())))
                count_val = word.lower().count(char)
                gold = str(count_val)
                expr = f'"{word}".lower().count("{char}")'

                phrasings = [
                    f"How many {char}'s are in the word '{word}'?",
                    f"How many {char}'s in {word}?",
                    f"how many {char}'s in {word}?",
                    f"how many {char}s in {word}?",
                    f"How many times does {char} appear in {word}?",
                    f"Count the letter {char} in {word}",
                    f"Count '{char}' in '{word}'",
                    f"How many '{char}' characters are in '{word}'?"
                ]
                q = rng.choice(phrasings)

            elif mode == 3:
                # String Reversal / Palindrome Check
                word = rng.choice(sample_words)
                gold = word[::-1]
                expr = f'"{word}"[::-1]'

                phrasings = [
                    f"What is the reverse of '{word}'?",
                    f"Reverse the string '{word}'",
                    f"Reverse {word}",
                    f"Spell {word} backwards",
                    f"What is {word} spelled backwards?",
                    f"What is the reverse of {word}?"
                ]
                q = rng.choice(phrasings)

            else:
                # Length and Case conversions
                word = rng.choice(sample_words)
                sub_op = rng.choice(["len", "upper", "lower"])
                if sub_op == "len":
                    gold = str(len(word))
                    expr = f'len("{word}")'
                    phrasings = [
                        f"What is the length of '{word}'?",
                        f"How many characters are in '{word}'?",
                        f"Length of {word}",
                        f"How long is the word {word}?"
                    ]
                elif sub_op == "upper":
                    gold = word.upper()
                    expr = f'"{word}".upper()'
                    phrasings = [
                        f"Convert '{word}' to uppercase",
                        f"Uppercase {word}",
                        f"What is {word} in all caps?"
                    ]
                else:
                    gold = word.lower()
                    expr = f'"{word}".lower()'
                    phrasings = [
                        f"Convert '{word}' to lowercase",
                        f"Lowercase {word}"
                    ]
                q = rng.choice(phrasings)

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

    def generate_suite_i_tasks(self, world: World, count: int, rng: random.Random) -> List[Task]:
        """Suite I: Active Counterfactual Inversion Benchmark (Real-world prior collision probes)."""
        tasks = []
        kb_samples = list(COUNTERFACTUAL_KNOWLEDGE_BASE)
        rng.shuffle(kb_samples)

        for i, item in enumerate(kb_samples[:count]):
            subj_name, ent_type, rel, cf_val, real_prior, doc_title, doc_text, q_template = item
            
            # Ensure entity exists in world
            ent_id = f"E_CF_{world.world_id}_{i+1}"
            world.entities[ent_id] = Entity(id=ent_id, name=subj_name, entity_type=ent_type)

            # Ensure fact exists
            fact_id = f"F_CF_{world.world_id}_{i+1}"
            world.facts[fact_id] = Fact(id=fact_id, subject_id=ent_id, relation=rel, value=cf_val, is_contingent=True)

            # Ensure document exists in world
            doc_id = f"D_CF_{world.world_id}_{i+1}"
            doc = Document(
                id=doc_id,
                title=doc_title,
                doc_type="registry",
                lines=[
                    DocumentLine(line_number=1, text=doc_text, fact_ids=[fact_id]),
                    DocumentLine(line_number=2, text=f"Official territorial classification for {subj_name}.", fact_ids=[])
                ]
            )
            world.documents[doc_id] = doc

            pg = ProofGraph(goal=f"counterfactual_grounding: {subj_name} {rel}")
            pg.required_document_lines[doc_id] = [1]

            tasks.append(Task(
                task_id=f"task_i_cf_{world.world_id}_{i+1}",
                task_type="counterfactual_inversion",
                suite="suite_i_counterfactual_inversion",
                question=q_template,
                gold_answer=str(cf_val),
                proof_graph=pg,
                world_id=world.world_id,
                is_retrieval_required=True,
                is_contingent=True,
                metadata={"prior_answer": real_prior, "target_entity": subj_name}
            ))

        return tasks

    def generate_anti_memorization_tasks(
        self,
        world: World,
        rng: random.Random,
        include_closed_book: bool = True,
    ) -> List[Task]:
        """Generates permutation, prior-reversal, evidence-disabled, and closed-book tasks."""
        tasks = []
        
        # 1. Closed-book leakage probe (must answer insufficient_evidence)
        if include_closed_book:
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

    def generate_all_tasks(
        self,
        world: World,
        rng: random.Random,
        include_counterfactual: bool = True,
        include_closed_book: bool = True,
    ) -> List[Task]:
        """Generates all task types for a given world."""
        all_tasks = []
        all_tasks.extend(self.generate_suite_a_tasks(world, count=2, rng=rng))
        all_tasks.extend(self.generate_suite_b_tasks(world, count=2, rng=rng))
        all_tasks.extend(self.generate_suite_c_tasks(world, count=3, rng=rng))
        all_tasks.extend(self.generate_suite_d_tasks(world, count=2, rng=rng))
        all_tasks.extend(self.generate_suite_e_tasks(world, count=2, rng=rng))
        all_tasks.extend(self.generate_suite_f_tasks(world, count=2, rng=rng))
        all_tasks.extend(self.generate_suite_g_tasks(world, count=1, rng=rng))
        all_tasks.extend(self.generate_suite_h_tasks(world, count=3, rng=rng))
        if include_counterfactual:
            all_tasks.extend(self.generate_suite_i_tasks(world, count=2, rng=rng))
        all_tasks.extend(
            self.generate_anti_memorization_tasks(world, rng=rng, include_closed_book=include_closed_book)
        )
        return all_tasks
