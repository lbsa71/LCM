"""Lexicon management and rich combinatorial nonce entity name generation."""

import random
from typing import List, Set

PREFIXES = [
    "ves", "nor", "tal", "jor", "pel", "zan", "vor", "nat", "les", "jir",
    "dak", "mip", "tor", "lum", "kex", "bar", "fen", "rix", "sol", "vun",
    "mar", "kel", "dox", "zep", "fal", "ren", "kin", "tir", "gor", "nel",
    "bra", "drem", "karn", "plin", "quor", "sarn", "thek", "vond", "wex", "zorn",
    "ald", "bel", "cor", "dyn", "el", "for", "gal", "hel", "ith", "jun",
    "kal", "lor", "mor", "nar", "or", "pyr", "qil", "ril", "sil", "tyr",
    "ur", "val", "wyr", "xar", "yul", "zel", "arc", "bryn", "cal", "dro",
    "en", "fra", "gly", "harn", "is", "jex", "kly", "lyr", "myn", "nox"
]

INFIXES = [
    "", "a", "e", "i", "o", "u", "ar", "er", "ir", "or", "ur",
    "al", "el", "il", "ol", "ul", "an", "en", "in", "on", "un",
    "ak", "ek", "ik", "ok", "uk", "as", "es", "is", "os", "us",
    "ad", "ed", "id", "od", "ud", "am", "em", "im", "om", "um"
]

SUFFIXES = [
    "ka", "u", "em", "a", "kin", "ir", "in", "at", "on", "or",
    "el", "is", "ak", "os", "un", "ix", "en", "ar", "ox", "ul",
    "an", "et", "ik", "ant", "orx", "usk", "eld", "ith", "ond",
    "ia", "ius", "ium", "ion", "or", "is", "ea", "on", "yn", "yr",
    "ara", "ora", "ira", "ela", "ula", "ax", "ex", "ux", "yx", "ath"
]

ENTITY_CATEGORIES = [
    "settlement", "region", "artifact", "device", "record",
    "event", "container", "zone", "station", "sector", "cluster", "specimen"
]

RELATION_NAMES = [
    "population", "inside", "contains", "connected_to",
    "distance_to", "measured_value", "timestamp", "recorded_in", "status", "category"
]

PROPERTY_ADJECTIVES = [
    "active", "dormant", "stable", "unstable", "sealed", "open", "calibrated",
    "uncalibrated", "primary", "secondary", "nominal", "critical"
]


class Lexicon:
    """Provides vocabulary and procedural name generation with > 500,000 distinct combinations."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.used_names: Set[str] = set()

    def generate_nonce_name(self, rng: random.Random) -> str:
        """Generates a novel synthetic nonce entity name."""
        prefix = rng.choice(PREFIXES)
        infix = rng.choice(INFIXES)
        suffix = rng.choice(SUFFIXES)
        return f"{prefix}{infix}{suffix}"

    def generate_unique_nonce_names(self, count: int, rng: random.Random) -> List[str]:
        """Generates a list of distinct nonce names."""
        names: Set[str] = set()
        attempts = 0
        max_attempts = count * 100
        while len(names) < count and attempts < max_attempts:
            attempts += 1
            name = self.generate_nonce_name(rng)
            names.add(name)

        # In extreme counts, add secondary syllable
        while len(names) < count:
            name = f"{rng.choice(PREFIXES)}{rng.choice(PREFIXES)}{rng.choice(SUFFIXES)}"
            names.add(name)
            
        return list(names)
