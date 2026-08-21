"""Counterfactual paired example generation (PRD Section 10)."""

import random
from typing import Dict, List, Tuple
from synth.ontology import Fact, Entity, Task, ProofGraph


class CounterfactualGenerator:
    """Generates strictly balanced paired counterfactual examples."""

    def __init__(self):
        pass

    def generate_pair(self, entity_a: str, entity_b: str, relation_type: str, rng: random.Random) -> Tuple[Dict, Dict]:
        """Generates a Pair A (True/Yes) and Pair B (False/No or Inverted) with identical surface structure."""
        if relation_type == "spatial_direction":
            dir_1 = "north"
            dir_2 = "south"
            
            # Pair A
            fact_a = f"The {entity_a} is {dir_1} of the {entity_b}."
            q = f"Is the {entity_a} {dir_1} of the {entity_b}?"
            pair_a = {
                "context": fact_a,
                "question": q,
                "answer": "yes",
                "relation": relation_type,
                "label": True
            }
            
            # Pair B
            fact_b = f"The {entity_a} is {dir_2} of the {entity_b}."
            pair_b = {
                "context": fact_b,
                "question": q,
                "answer": "no",
                "relation": relation_type,
                "label": False
            }
            return pair_a, pair_b

        elif relation_type == "numeric_comparison":
            val_high = rng.randint(500, 999)
            val_low = rng.randint(100, 499)
            
            q = f"Does {entity_a} have a greater value than {entity_b}?"
            
            # Pair A: A > B
            fact_a = f"{entity_a} has a value of {val_high}. {entity_b} has a value of {val_low}."
            pair_a = {
                "context": fact_a,
                "question": q,
                "answer": "yes",
                "relation": relation_type,
                "label": True
            }
            
            # Pair B: A < B
            fact_b = f"{entity_a} has a value of {val_low}. {entity_b} has a value of {val_high}."
            pair_b = {
                "context": fact_b,
                "question": q,
                "answer": "no",
                "relation": relation_type,
                "label": False
            }
            return pair_a, pair_b

        elif relation_type == "containment":
            # Pair A: A inside B
            fact_a = f"The entity {entity_a} is inside {entity_b}."
            q = f"Is {entity_a} located inside {entity_b}?"
            pair_a = {
                "context": fact_a,
                "question": q,
                "answer": "yes",
                "relation": relation_type,
                "label": True
            }
            
            # Pair B: A inside other
            other_c = f"zone_{rng.randint(10, 99)}"
            fact_b = f"The entity {entity_a} is inside {other_c}."
            pair_b = {
                "context": fact_b,
                "question": q,
                "answer": "no",
                "relation": relation_type,
                "label": False
            }
            return pair_a, pair_b

        else:
            # Status pair
            q = f"Is {entity_a} in active status?"
            pair_a = {
                "context": f"{entity_a} status is active.",
                "question": q,
                "answer": "yes",
                "relation": "status",
                "label": True
            }
            pair_b = {
                "context": f"{entity_a} status is dormant.",
                "question": q,
                "answer": "no",
                "relation": "status",
                "label": False
            }
            return pair_a, pair_b
