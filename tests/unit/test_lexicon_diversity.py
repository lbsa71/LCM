"""Unit test verifying high combinatorial lexical diversity."""

import random
import pytest

from synth.language.lexicon import Lexicon


def test_combinatorial_lexicon_diversity():
    """Verify that lexicon can easily produce 10,000 distinct names without fallback obj_ indices."""
    lex = Lexicon(seed=42)
    rng = random.Random(42)
    names = lex.generate_unique_nonce_names(10000, rng)
    
    assert len(names) == 10000
    assert len(set(names)) == 10000
    # Ensure standard pronounceable names are generated, not just obj_ fallbacks
    obj_count = sum(1 for n in names if n.startswith("obj_"))
    assert obj_count == 0, f"Encountered {obj_count} obj_ fallback names"
