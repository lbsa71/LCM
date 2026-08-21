"""Deterministic BM25 lexical search engine over synthetic world documents."""

import math
import re
from typing import Any, Dict, List, Optional
from synth.ontology import World, Document


def tokenize(text: str) -> List[str]:
    """Deterministic tokenization."""
    return re.findall(r"\b\w+\b", text.lower())


class DeterministicBM25Search:
    """Deterministic BM25 document search with stable tie-breaking."""

    def __init__(self, world: World, k1: float = 1.5, b: float = 0.75):
        self.world = world
        self.k1 = k1
        self.b = b
        self.doc_ids = sorted(list(world.documents.keys()))
        self.docs = [world.documents[d_id] for d_id in self.doc_ids]
        
        # Build index
        self.doc_lengths = []
        self.doc_tokens = []
        self.doc_freqs: Dict[str, int] = {}
        
        for doc in self.docs:
            full_text = f"{doc.title} " + " ".join([line.text for line in doc.lines])
            toks = tokenize(full_text)
            self.doc_tokens.append(toks)
            self.doc_lengths.append(len(toks))
            
            seen_terms = set(toks)
            for term in seen_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.N = len(self.docs)
        self.avgdl = sum(self.doc_lengths) / max(1, self.N)

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Executes deterministic BM25 search."""
        q_tokens = tokenize(query)
        if not q_tokens or self.N == 0:
            return {"status": "success", "results": []}

        scores = []
        for idx, doc_id in enumerate(self.doc_ids):
            tokens = self.doc_tokens[idx]
            doc_len = self.doc_lengths[idx]
            
            # Calculate term frequencies
            tf_dict: Dict[str, int] = {}
            for t in tokens:
                tf_dict[t] = tf_dict.get(t, 0) + 1

            score = 0.0
            for qt in q_tokens:
                if qt not in tf_dict:
                    continue
                df = self.doc_freqs.get(qt, 0)
                # BM25 IDF
                idf = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
                tf = tf_dict[qt]
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                score += idf * (num / max(1e-6, denom))

            if score > 0:
                doc = self.docs[idx]
                first_line = doc.lines[0].text if doc.lines else ""
                scores.append({
                    "document_id": doc_id,
                    "score": round(score, 4),
                    "title": doc.title,
                    "snippet": first_line
                })

        # Stable sort: score descending, document_id ascending
        scores.sort(key=lambda x: (-x["score"], x["document_id"]))
        return {
            "status": "success",
            "results": scores[:limit]
        }
