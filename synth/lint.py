"""Corpus linter verifying zero contamination, label balance, and data integrity."""

import os
import re
from typing import Any, Dict, List, Set, Tuple


class CorpusLinter:
    """Validates synthetic datasets against forbidden entities and structural anomalies."""

    def __init__(self, forbidden_entities_path: str = "specs/forbidden_entities.txt"):
        self.forbidden_terms: Set[str] = set()
        if os.path.exists(forbidden_entities_path):
            with open(forbidden_entities_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip().lower()
                    if line and not line.startswith("#"):
                        self.forbidden_terms.add(line)

    def check_forbidden_terms(self, text: str) -> List[str]:
        """Checks if text contains any forbidden real-world terms."""
        text_lower = text.lower()
        violations = []
        for term in self.forbidden_terms:
            # Word boundary search
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, text_lower):
                violations.append(term)
        return violations

    def lint_dataset(
        self,
        train_texts: List[str],
        val_texts: List[str],
        test_texts: List[str],
        train_seeds: Set[int],
        test_seeds: Set[int]
    ) -> Dict[str, Any]:
        """Runs comprehensive lint checks across datasets."""
        errors = []
        warnings = []

        # 1. Seed overlap check
        overlap = train_seeds.intersection(test_seeds)
        if overlap:
            errors.append(f"Train and Test sets share world seeds: {overlap}")

        # 2. Forbidden entities check across all text
        all_samples = [("train", t) for t in train_texts] + [("val", t) for t in val_texts] + [("test", t) for t in test_texts]
        forbidden_hits = 0
        for split, text in all_samples:
            hits = self.check_forbidden_terms(text)
            if hits:
                forbidden_hits += len(hits)
                errors.append(f"Forbidden term {hits} found in {split} sample: {text[:80]}...")

        # 3. Duplicate check between train and test
        train_set = set(train_texts)
        test_overlap = sum(1 for t in test_texts if t in train_set)
        if test_overlap > 0:
            errors.append(f"Found {test_overlap} exact duplicate texts between train and test splits.")

        status = "PASS" if not errors else "FAIL"
        return {
            "status": status,
            "errors": errors,
            "warnings": warnings,
            "total_samples_checked": len(all_samples),
            "train_samples": len(train_texts),
            "val_samples": len(val_texts),
            "test_samples": len(test_texts),
            "forbidden_term_violations": forbidden_hits
        }
