"""Tabular EvidenceProvider adapter for relational records and SQL-like filtering."""

from typing import Any, Dict, List, Optional
from agent.adapters.base import EvidenceProvider


class TabularEvidenceProvider(EvidenceProvider):
    """EvidenceProvider for structured/relational table records."""

    def __init__(self, table_name: str, schema: Dict[str, str], records: Optional[List[Dict[str, Any]]] = None):
        self.table_name = table_name
        self.table_schema = schema
        self.records = records or []

    def introspect(self) -> Dict[str, Any]:
        return {
            "type": "tabular",
            "table_name": self.table_name,
            "schema": self.table_schema,
            "capabilities": ["filter", "scan"],
            "total_records": len(self.records),
        }

    def filter(self, field: str, op: str, value: Any) -> List[Dict[str, Any]]:
        """Filters in-memory records with predicate operators."""
        op_upper = op.upper()
        results = []
        for row in self.records:
            if field not in row:
                continue
            row_val = row[field]
            
            if op_upper == "EQ" and row_val == value:
                results.append(row)
            elif op_upper == "GT" and row_val > value:
                results.append(row)
            elif op_upper == "LT" and row_val < value:
                results.append(row)
            elif op_upper == "CONTAINS" and str(value).lower() in str(row_val).lower():
                results.append(row)
        return results

    def scan(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Scans records up to limit."""
        return self.records[:limit]
