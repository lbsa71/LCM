"""Corpus manifest generator and cryptographic checksum calculator."""

import hashlib
import json
import os
from typing import Any, Dict


def compute_file_sha256(file_path: str) -> str:
    """Computes SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_manifest(
    run_dir: str,
    config: Dict[str, Any],
    stats: Dict[str, Any],
    validation_status: str
) -> Dict[str, Any]:
    """Generates and writes corpus_manifest.json with hashes and statistics."""
    file_hashes = {}
    for root, _, files in os.walk(run_dir):
        for file in files:
            if file == "corpus_manifest.json":
                continue
            full_p = os.path.join(root, file)
            rel_p = os.path.relpath(full_p, run_dir)
            file_hashes[rel_p] = compute_file_sha256(full_p)

    manifest = {
        "generator_version": "0.1.0",
        "preset": config.get("name", "unknown"),
        "seed": config.get("seed", 42),
        "validation_status": validation_status,
        "statistics": stats,
        "file_hashes": file_hashes
    }

    manifest_path = os.path.join(run_dir, "corpus_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
