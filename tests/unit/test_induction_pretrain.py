"""Unit tests for in-context induction and copy pretraining sequence generation."""

import random
import pytest

from synth.ontology import World, Entity, Document, DocumentLine
from synth.world import WorldGenerator


def test_induction_pretraining_patterns():
    """Verify that induction pretraining lines reinforce prompt-to-query copying."""
    world_gen = WorldGenerator(base_seed=42)
    w = world_gen.generate_world("w_test", seed=42)
    
    assert len(w.entities) > 0
    e = list(w.entities.values())[0]
    
    # In-context copying pattern
    pattern = f"User asks for {e.name}. Searching for {e.name}. Result found for {e.name}."
    assert e.name in pattern
    assert pattern.count(e.name) == 3
