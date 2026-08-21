"""Lexicon management and nonce entity name generation."""

import random
from typing import List, Set

PREFIXES = [
    "ves", "nor", "tal", "jor", "pel", "zan", "vor", "nat", "les", "jir",
    "dak", "mip", "tor", "lum", "kex", "bar", "fen", "rix", "sol", "vun",
    "mar", "kel", "dox", "zep", "fal", "ren", "kin", "tir", "gor", "nel",
    "bra", "drem", "karn", "plin", "quor", "sarn", "thek", "vond", "wex", "zorn"
]

SUFFIXES = [
    "ka", "u", "em", "a", "kin", "ir", "in", "at", "on", "or",
    "el", "is", "ak", "os", "un", "ix", "en", "ar", "ox", "ul",
    "an", "et", "or", "ik", "ant", "orx", "usk", "eld", "ith", "ond"
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
    """Provides vocabulary and procedural name generation."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.used_names: Set[str] = set()

    def generate_nonce_name(self, rng: random.Random) -> str:
        """Generates a novel synthetic nonce entity name."""
        prefix = rng.choice(PREFIXES)
        suffix = rng.choice(SUFFIXES)
        name = f"{prefix}{suffix}"
        return name

    def generate_unique_nonce_names(self, count: int, rng: random.Random) -> List[str]:
        """Generates a list of distinct nonce names."""
        names: Set[str] = set()
        attempts = 0
        while len(names) < count and attempts < count * 50:
            attempts += 1
            name = self.generate_nonce_name(rng)
            names.add(name)
        # If we need more, compose double-syllable or indexed names
        idx = 1
        while len(names) < count:
            name = f"obj_{rng.choice(PREFIXES)}{idx}"
            names.add(name)
            idx += 1
        return list(names)
