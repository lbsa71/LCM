"""Line-addressed document reader tool (PRD Section 25.2)."""

from typing import Any, Dict
from synth.ontology import World


class DocumentReader:
    """Retrieves document text formatted with deterministic line numbers."""

    def __init__(self, world: World):
        self.world = world

    def read(self, document_id: str) -> Dict[str, Any]:
        """Reads a document by its ID."""
        doc = self.world.documents.get(document_id)
        if not doc:
            return {
                "status": "error",
                "error_type": "DOCUMENT_NOT_FOUND",
                "message": f"Document with ID '{document_id}' was not found in the world environment."
            }

        return {
            "status": "success",
            "document_id": doc.id,
            "title": doc.title,
            "text": doc.formatted_text
        }
