"""
Simple schema linking for Spider-style Text-to-SQL.

Goal:
- Given (question, db_id), select a small set of relevant tables/columns
  to include in the prompt (RAG-style schema retrieval).

Design constraints:
- Pure Python (no heavy external deps).
- Robust to missing/odd schemas: never crash.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


_ALNUM_RE = re.compile(r"[A-Za-z0-9]+")
_CAMEL_RE = re.compile(r"([a-z])([A-Z])")


def _normalize_identifier(text: str) -> str:
    """
    Normalize a schema identifier:
    - split underscores
    - split camelCase / PascalCase boundaries
    - lowercase
    """
    text = str(text or "")
    text = text.replace("_", " ")
    text = _CAMEL_RE.sub(r"\1 \2", text)
    return text.lower()


def _tokenize(text: str) -> List[str]:
    text = _normalize_identifier(text)
    return _ALNUM_RE.findall(text)


@dataclass(frozen=True)
class TableSchema:
    table_name: str
    columns: Tuple[str, ...]


class SchemaLinker:
    """
    Loads Spider `tables.json` and (optionally) SQLite schemas from disk.
    Provides a lightweight table scoring function based on token overlap.
    """

    def __init__(self, tables_json_path: str, db_root: Optional[str] = None):
        self.tables_json_path = tables_json_path
        self.db_root = db_root
        self._tables_by_db: Dict[str, List[TableSchema]] = {}
        self._sqlite_schema_cache: Dict[str, Dict[str, List[str]]] = {}
        self._load_tables_json()

    def _load_tables_json(self) -> None:
        with open(self.tables_json_path) as f:
            entries = json.load(f)

        tables_by_db: Dict[str, List[TableSchema]] = {}
        for entry in entries:
            db_id = entry["db_id"]
            table_names: List[str] = entry.get("table_names_original") or entry.get("table_names") or []
            col_names: List[Sequence] = entry.get("column_names_original") or entry.get("column_names") or []

            columns_by_table_idx: Dict[int, List[str]] = {i: [] for i in range(len(table_names))}
            for col in col_names:
                # Spider format: [table_idx, col_name]
                if not col or len(col) < 2:
                    continue
                table_idx, col_name = col[0], col[1]
                if table_idx is None or table_idx < 0:
                    continue  # skip "*"
                if table_idx not in columns_by_table_idx:
                    continue
                columns_by_table_idx[table_idx].append(str(col_name))

            tables: List[TableSchema] = []
            for i, tname in enumerate(table_names):
                cols = tuple(columns_by_table_idx.get(i, []))
                tables.append(TableSchema(table_name=str(tname), columns=cols))

            tables_by_db[db_id] = tables

        self._tables_by_db = tables_by_db

    def _db_path(self, db_id: str) -> Optional[str]:
        if not self.db_root:
            return None
        path = os.path.join(self.db_root, db_id, f"{db_id}.sqlite")
        return path if os.path.exists(path) else None

    def _load_sqlite_schema(self, db_id: str) -> Dict[str, List[str]]:
        """
        Load actual SQLite schema (table -> columns). Cached per db_id.
        """
        if db_id in self._sqlite_schema_cache:
            return self._sqlite_schema_cache[db_id]

        schema: Dict[str, List[str]] = {}
        db_path = self._db_path(db_id)
        if not db_path:
            self._sqlite_schema_cache[db_id] = schema
            return schema

        try:
            with closing(sqlite3.connect(db_path)) as conn:
                cursor = conn.cursor()
                tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
                for (table_name,) in tables:
                    columns = cursor.execute(f"PRAGMA table_info({table_name});").fetchall()
                    schema[str(table_name)] = [str(col[1]) for col in columns]
        except Exception:
            schema = {}

        self._sqlite_schema_cache[db_id] = schema
        return schema

    def get_schema(self, db_id: str) -> List[TableSchema]:
        """
        Returns a list of table schemas for this db.
        Prefers `tables.json` (Spider canonical), but can fallback to SQLite if needed.
        """
        tables = self._tables_by_db.get(db_id, [])
        if tables:
            return tables

        sqlite_schema = self._load_sqlite_schema(db_id)
        return [TableSchema(table_name=t, columns=tuple(cols)) for t, cols in sqlite_schema.items()]

    def score_tables(self, question: str, db_id: str) -> List[Tuple[float, TableSchema]]:
        """
        Score each table using token overlap:
        - table token overlap (higher weight)
        - column token overlap (lower weight)
        """
        q_tokens = set(_tokenize(question))
        tables = self.get_schema(db_id)

        scored: List[Tuple[float, TableSchema]] = []
        for t in tables:
            table_tokens = set(_tokenize(t.table_name))
            col_tokens: set[str] = set()
            for c in t.columns:
                col_tokens.update(_tokenize(c))

            table_overlap = len(q_tokens & table_tokens)
            col_overlap = len(q_tokens & col_tokens)

            # Simple weighted overlap (tuned to bias table matches).
            score = 3.0 * table_overlap + 1.0 * col_overlap

            # Small boost for substring mentions (helps e.g. "album" vs "albums").
            q_text = _normalize_identifier(question)
            if t.table_name and _normalize_identifier(t.table_name) in q_text:
                score += 0.5

            scored.append((score, t))

        scored.sort(key=lambda x: (x[0], x[1].table_name), reverse=True)
        return scored

    def select_top_tables(self, question: str, db_id: str, top_k: int = 4) -> List[TableSchema]:
        scored = self.score_tables(question, db_id)
        if not scored:
            return []
        top_k = max(1, int(top_k))
        selected = [t for _, t in scored[:top_k]]

        # If everything scores 0, still return a stable selection.
        if scored[0][0] <= 0:
            tables = self.get_schema(db_id)
            return tables[:top_k]

        return selected

    def columns_for_selected_tables(self, db_id: str, selected_tables: Iterable[TableSchema]) -> Dict[str, List[str]]:
        """
        Returns only columns belonging to selected tables.
        Prefer SQLite columns (actual DB) if available; fallback to tables.json.
        """
        sqlite_schema = self._load_sqlite_schema(db_id)
        out: Dict[str, List[str]] = {}
        for t in selected_tables:
            if t.table_name in sqlite_schema and sqlite_schema[t.table_name]:
                out[t.table_name] = sqlite_schema[t.table_name]
            else:
                out[t.table_name] = list(t.columns)
        return out

    def format_relevant_schema(self, question: str, db_id: str, top_k: int = 4) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Returns:
        - lines: ["table(col1, col2)", ...]
        - selected: {table: [cols...], ...}
        """
        selected_tables = self.select_top_tables(question, db_id, top_k=top_k)
        selected = self.columns_for_selected_tables(db_id, selected_tables)

        lines: List[str] = []
        for table_name, cols in selected.items():
            cols_str = ", ".join(cols)
            lines.append(f"{table_name}({cols_str})")

        return lines, selected

