"""Grammar rules and sentence variations for transforming facts into natural language."""

from typing import Any, List, Optional
import random

from synth.ontology import Fact, Entity


class GrammarRenderer:
    """Renders structured semantic facts into varied surface linguistic forms."""

    def __init__(self, template_set: str = "all"):
        self.template_set = template_set  # "train", "eval", "all"

    def render_fact(self, fact: Fact, entity_map: dict[str, Entity], rng: random.Random) -> str:
        """Renders a fact into a varied natural language sentence."""
        subj = entity_map.get(fact.subject_id)
        subj_name = subj.name if subj else fact.subject_id
        subj_cat = subj.entity_type if subj else "entity"
        rel = fact.relation
        val = fact.value

        templates = []

        if rel == "population":
            templates = [
                f"The {subj_cat} {subj_name} has a recorded population of {val}.",
                f"Recorded for the {subj_cat} {subj_name} is a population of {val}.",
                f"A population count of {val} is documented for {subj_name}.",
                f"The population of {subj_name} is {val}.",
                f"{subj_name}, a {subj_cat}, holds a population of {val}.",
                f"Census records indicate that {subj_name} has {val} inhabitants.",
                f"The recorded headcount for {subj_name} is currently {val}."
            ]
        elif rel == "inside":
            obj = entity_map.get(str(val))
            obj_name = obj.name if obj else str(val)
            obj_cat = obj.entity_type if obj else "region"
            templates = [
                f"The {subj_cat} {subj_name} is located inside {obj_name}.",
                f"Inside {obj_name} lies the {subj_cat} {subj_name}.",
                f"{subj_name} is situated within the boundaries of {obj_name}.",
                f"Geographic logs confirm that {subj_name} is inside {obj_name}.",
                f"The {obj_cat} {obj_name} contains the {subj_cat} {subj_name}.",
                f"Positioned in {obj_name} is the {subj_cat} {subj_name}."
            ]
        elif rel == "measured_value":
            templates = [
                f"The measured reading for {subj_name} is {val} units.",
                f"A value of {val} was measured for {subj_name}.",
                f"Sensor instruments record {val} units for {subj_name}.",
                f"For the {subj_cat} {subj_name}, the measurement shows {val}."
            ]
        elif rel == "status":
            templates = [
                f"The current operating status of {subj_name} is {val}.",
                f"System telemetry reports that {subj_name} is {val}.",
                f"{subj_name} is marked as {val}.",
                f"Status report: {subj_name} is {val}."
            ]
        elif rel == "distance_to":
            meta = fact.metadata
            target = meta.get("target_name", "target")
            templates = [
                f"The distance between {subj_name} and {target} is {val} leagues.",
                f"Separating {subj_name} and {target} is a distance of {val} leagues.",
                f"Traversing from {subj_name} to {target} spans {val} leagues."
            ]
        elif rel == "timestamp":
            templates = [
                f"The event {subj_name} occurred at cycle {val}.",
                f"At cycle {val}, the event {subj_name} was logged.",
                f"Chronological records show that {subj_name} took place at cycle {val}."
            ]
        elif rel == "category":
            templates = [
                f"The entity {subj_name} belongs to the category {val}.",
                f"{subj_name} is classified as a {val}.",
                f"Classification records identify {subj_name} as a {val}."
            ]
        else:
            templates = [
                f"The property {rel} of {subj_name} is {val}.",
                f"For {subj_name}, the {rel} is recorded as {val}."
            ]

        # Holdout partitioning if specified
        if self.template_set == "train":
            # Select from first 75% of templates
            cutoff = max(1, int(len(templates) * 0.75))
            chosen_templates = templates[:cutoff]
        elif self.template_set == "eval":
            # Select from last 25% of templates (surface form holdout)
            cutoff = max(1, int(len(templates) * 0.75))
            chosen_templates = templates[cutoff:] if len(templates) > 1 else templates
        else:
            chosen_templates = templates

        return rng.choice(chosen_templates)

    def render_question(self, fact: Fact, entity_map: dict[str, Entity], rng: random.Random) -> str:
        """Renders a natural language query for a given fact."""
        subj = entity_map.get(fact.subject_id)
        subj_name = subj.name if subj else fact.subject_id
        rel = fact.relation

        if rel == "population":
            q_forms = [
                f"What is the recorded population of {subj_name}?",
                f"What is the population of {subj_name}?",
                f"How many inhabitants are recorded for {subj_name}?",
                f"Retrieve the population count of {subj_name}."
            ]
        elif rel == "inside":
            q_forms = [
                f"Which region contains {subj_name}?",
                f"Where is {subj_name} located?",
                f"In which zone or region is {subj_name} situated?",
                f"What contains the entity {subj_name}?"
            ]
        elif rel == "measured_value":
            q_forms = [
                f"What is the measured value for {subj_name}?",
                f"What reading is recorded for {subj_name}?",
                f"Retrieve the measurement recorded for {subj_name}."
            ]
        elif rel == "status":
            q_forms = [
                f"What is the current status of {subj_name}?",
                f"Is {subj_name} active or dormant?",
                f"Retrieve the operating status of {subj_name}."
            ]
        elif rel == "timestamp":
            q_forms = [
                f"At what cycle did {subj_name} occur?",
                f"When did the event {subj_name} take place?",
                f"Retrieve the timestamp for {subj_name}."
            ]
        else:
            q_forms = [
                f"What is the {rel} of {subj_name}?",
                f"Retrieve the {rel} for {subj_name}."
            ]

        return rng.choice(q_forms)
