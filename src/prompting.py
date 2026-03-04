from __future__ import annotations

import os
import re
import sqlite3
from contextlib import closing
from typing import Dict, Optional

import torch

# Keep for compatibility with existing imports. Schema linking is disabled for
# SFT/RL alignment in this project version (full schema, deterministic order).
USE_SCHEMA_LINKING = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_ROOT = os.path.join(PROJECT_ROOT, "data", "database")

SCHEMA_CACHE: Dict[str, str] = {}


def get_schema_text(db_id: str) -> str:
    """
    Deterministic schema string:
      table(col1, col2, ...)
    Tables ordered alphabetically. Columns kept in PRAGMA order.
    """
    if db_id in SCHEMA_CACHE:
        return SCHEMA_CACHE[db_id]

    db_path = os.path.join(DB_ROOT, db_id, f"{db_id}.sqlite")
    schema_lines = []
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            cur = conn.cursor()
            tables = cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
            table_names = sorted([t[0] for t in tables if t and isinstance(t[0], str)])
            for tname in table_names:
                cols = cur.execute(f'PRAGMA table_info("{tname}")').fetchall()
                col_names = [c[1] for c in cols if c and isinstance(c[1], str)]
                schema_lines.append(f"{tname}({', '.join(col_names)})")
    except Exception:
        schema_lines = []

    schema_text = "\n".join(schema_lines).strip()
    SCHEMA_CACHE[db_id] = schema_text
    return schema_text


def clean_gold_sql(sql: str) -> str:
    """
    Lowercase SQL + strip common Spider aliases safely.
    If alias removal is ambiguous (same table used multiple times), keep SQL as-is.
    """
    if not isinstance(sql, str):
        return ""
    s = sql.strip().rstrip(";").strip()
    if not s:
        return ""

    # Attempt to resolve T1/T2 aliases to table names for simple cases.
    # Build alias -> table map from FROM/JOIN clauses.
    alias_map: Dict[str, str] = {}
    table_counts: Dict[str, int] = {}

    for m in re.finditer(r"\b(from|join)\s+([a-zA-Z_][\w$]*)\s+(?:as\s+)?(t\d+)\b", s, flags=re.I):
        table = m.group(2)
        alias = m.group(3)
        table_counts[table.lower()] = table_counts.get(table.lower(), 0) + 1
        alias_map[alias.lower()] = table

    # If any table appears multiple times, alias removal can be ambiguous → skip.
    if any(c > 1 for c in table_counts.values()):
        return s.lower()

    # Replace alias-qualified refs alias.col -> table.col
    out = s
    for alias, table in alias_map.items():
        out = re.sub(rf"\b{re.escape(alias)}\.", f"{table}.", out, flags=re.I)

    # Remove alias declarations: "table AS t1" or "table t1"
    for alias, table in alias_map.items():
        out = re.sub(rf"\b{re.escape(table)}\s+as\s+{re.escape(alias)}\b", table, out, flags=re.I)
        out = re.sub(rf"\b{re.escape(table)}\s+{re.escape(alias)}\b", table, out, flags=re.I)

    return out.lower().strip()


def build_prompt(
    question: str,
    db_id: str,
    *,
    schema_text: str,
    training_sql: Optional[str] = None,
) -> str:
    """
    Required prompt format:

    You are a SQLite expert.

    Database: <db_id>

    Schema:
    <table>(col1, col2, ...)
    ...

    Question:
    <question>

    SQL:
    <gold sql>   (training only)
    """
    base = (
        "You are a SQLite expert.\n\n"
        f"Database: {db_id}\n\n"
        "Schema:\n"
        f"{schema_text}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "SQL:"
    )
    if training_sql is None:
        return base
    return base + "\n" + training_sql


def encode_prompt(
    tokenizer,
    question: str,
    db_id: str,
    *,
    device: str,
    max_input_tokens: int = 512,
    training_sql: Optional[str] = None,
) -> torch.Tensor:
    """
    Inference mode: stops at "SQL:"
    Training mode: can include SQL target (optional; we still recommend decoder labels).
    Truncation happens only on schema portion by character trimming (deterministic).
    """
    schema_text = get_schema_text(db_id)
    prompt = build_prompt(question, db_id, schema_text=schema_text, training_sql=training_sql)
    enc = tokenizer(
        prompt,
        truncation=True,
        max_length=max_input_tokens,
        padding=False,
        return_tensors="pt",
    )
    return enc.input_ids[0].to(device)
