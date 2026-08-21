"""Unit tests for world generator and counterfactual generation."""

import random
from synth.world import WorldGenerator
from synth.documents.generator import DocumentGenerator
from synth.language.counterfactual import CounterfactualGenerator


def test_seed_determinism():
    gen1 = WorldGenerator(base_seed=42)
    w1 = gen1.generate_world("w1", seed=100)

    gen2 = WorldGenerator(base_seed=42)
    w2 = gen2.generate_world("w1", seed=100)

    assert len(w1.entities) == len(w2.entities)
    assert len(w1.facts) == len(w2.facts)
    for e_id in w1.entities:
        assert w1.entities[e_id].name == w2.entities[e_id].name
    for f_id in w1.facts:
        assert w1.facts[f_id].value == w2.facts[f_id].value


def test_seed_permutation():
    gen = WorldGenerator(base_seed=42)
    w1 = gen.generate_world("w1", seed=101)
    w2 = gen.generate_world("w2", seed=202)

    # Names and fact values should differ
    names1 = {e.name for e in w1.entities.values()}
    names2 = {e.name for e in w2.entities.values()}
    assert names1 != names2


def test_counterfactual_pairing():
    cf_gen = CounterfactualGenerator()
    rng = random.Random(42)
    p_a, p_b = cf_gen.generate_pair("noru", "veska", "spatial_direction", rng)

    assert p_a["answer"] == "yes"
    assert p_b["answer"] == "no"
    assert p_a["question"] == p_b["question"]
