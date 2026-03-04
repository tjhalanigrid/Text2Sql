from __future__ import annotations

import os
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple, Union

try:
    import sqlparse
    from sqlparse.sql import Function, Identifier, IdentifierList, Statement, Token, Where
    from sqlparse.tokens import DML, Keyword, Name, Number, Punctuation, String, Whitespace
except Exception:  # pragma: no cover
    sqlparse = None  # type: ignore[assignment]
    Statement = object  # type: ignore[misc,assignment]
    Token = object  # type: ignore[misc,assignment]


def _normalize_sql(sql: str) -> str:
    if not isinstance(sql, str):
        return ""
    s = sql.strip()
    if s.startswith("```"):
        # Strip markdown fences if present.
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", s).strip()
        s = re.sub(r"\n?```$", "", s).strip()
    if s.lower().startswith("sql:"):
        s = s[4:].strip()
    # Keep only the first statement to avoid accidental multi-statement execution.
    if ";" in s:
        s = s.split(";", 1)[0].strip()
    return s


def _connect_readonly(db_path: str) -> sqlite3.Connection:
    # Read-only prevents any accidental mutation during reward computation.
    # Note: requires SQLite URI support (built-in).
    uri = f"file:{os.path.abspath(db_path)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.execute("PRAGMA query_only = ON;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _with_timeout(conn: sqlite3.Connection, timeout_s: float = 1.0) -> None:
    start = time.monotonic()

    def _handler() -> int:
        return 1 if (time.monotonic() - start) > timeout_s else 0

    # Call handler every N VM opcodes.
    conn.set_progress_handler(_handler, 10_000)


def _list_tables(conn: sqlite3.Connection) -> List[str]:
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        return [r[0] for r in cur.fetchall() if r and isinstance(r[0], str)]
    except sqlite3.Error:
        return []


def _contains_table_name(sql: str, table_names: Sequence[str]) -> bool:
    s = sql.lower()
    for t in table_names:
        tl = t.lower()
        if not tl:
            continue
        if re.search(rf"\b{re.escape(tl)}\b", s):
            return True
    return False


def _explain_query_plan(conn: sqlite3.Connection, sql: str) -> bool:
    try:
        _with_timeout(conn, timeout_s=1.0)
        conn.execute(f"EXPLAIN QUERY PLAN {sql}")
        return True
    except sqlite3.Error:
        return False


def _execute(conn: sqlite3.Connection, sql: str, max_rows: int = 1000) -> Tuple[bool, List[Tuple], Optional[str]]:
    try:
        _with_timeout(conn, timeout_s=1.0)
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows)
        # Normalize to plain tuples for deterministic comparison.
        norm_rows = [tuple(r) for r in rows]
        return True, norm_rows, None
    except sqlite3.Error as e:
        return False, [], str(e)


_SQL_KEYWORDS_TO_IGNORE = {
    "select",
    "from",
    "where",
    "join",
    "inner",
    "left",
    "right",
    "full",
    "outer",
    "on",
    "group",
    "by",
    "order",
    "limit",
    "having",
    "distinct",
    "union",
    "intersect",
    "except",
    "as",
    "and",
    "or",
    "not",
    "in",
    "is",
    "null",
    "like",
    "between",
    "case",
    "when",
    "then",
    "else",
    "end",
    "asc",
    "desc",
}

_SQL_FUNCTIONS_TO_IGNORE = {
    "count",
    "avg",
    "min",
    "max",
    "sum",
    "lower",
    "upper",
    "substr",
    "coalesce",
    "round",
    "date",
    "datetime",
    "strftime",
}


def extract_tables(sql: str) -> Set[str]:
    """
    Best-effort table extraction from SQL using sqlparse.
    Returns lowercase table names (unqualified).
    """
    sql = _normalize_sql(sql)
    if not sql:
        return set()
    if sqlparse is None:
        # Fallback: naive regex for FROM/JOIN.
        found = set()
        for m in re.finditer(r"\b(from|join)\s+([a-zA-Z_][\w$]*)", sql, flags=re.I):
            found.add(m.group(2).lower())
        return found

    try:
        statements = sqlparse.parse(sql)
    except Exception:
        return set()

    tables: Set[str] = set()

    def _add_identifier_as_table(ident: Identifier) -> None:
        # Prefer real name over alias; strip any schema prefix.
        name = ident.get_real_name() or ident.get_name()
        if not name:
            return
        tables.add(name.lower())

    for st in statements:
        if not isinstance(st, Statement):
            continue
        seen_from = False
        for tok in st.flatten():
            if tok.ttype in Whitespace:
                continue
            if tok.ttype is Keyword and tok.value.upper() in {"FROM", "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN"}:
                seen_from = True
                continue
            if not seen_from:
                continue

            if isinstance(tok, Identifier):
                _add_identifier_as_table(tok)
                seen_from = False
            elif tok.ttype is Name:
                tables.add(tok.value.lower())
                seen_from = False
            elif tok.ttype is Keyword and tok.value.upper() in {"WHERE", "GROUP", "ORDER", "HAVING", "LIMIT"}:
                seen_from = False

    return tables


def extract_columns(sql: str) -> Set[str]:
    """
    Best-effort column extraction from SQL using sqlparse.
    Returns lowercase column names (unqualified).
    """
    sql = _normalize_sql(sql)
    if not sql:
        return set()
    if sqlparse is None:
        # Fallback: naive dotted identifiers and bare names after SELECT/WHERE/etc.
        cols = set()
        for m in re.finditer(r"\b([a-zA-Z_][\w$]*)\b", sql):
            w = m.group(1).lower()
            if w in _SQL_KEYWORDS_TO_IGNORE or w in _SQL_FUNCTIONS_TO_IGNORE:
                continue
            cols.add(w)
        return cols

    try:
        statements = sqlparse.parse(sql)
    except Exception:
        return set()

    cols: Set[str] = set()

    def _maybe_add_col(name: Optional[str]) -> None:
        if not name:
            return
        n = name.strip().strip('"').strip("'").lower()
        if not n or n == "*":
            return
        if n in _SQL_KEYWORDS_TO_IGNORE or n in _SQL_FUNCTIONS_TO_IGNORE:
            return
        cols.add(n)

    def _handle_identifier(ident: Identifier) -> None:
        # If qualified (t.col), keep only col for overlap/hallucination checks.
        _maybe_add_col(ident.get_real_name() or ident.get_name())

    for st in statements:
        if not isinstance(st, Statement):
            continue
        for tok in st.flatten():
            # Skip whitespace/punctuation/string literals/numbers.
            if getattr(tok, "ttype", None) in (Whitespace, Punctuation, String, Number):
                continue

            if isinstance(tok, Function):
                fname = tok.get_name()
                if fname:
                    # Don't treat function name as a column.
                    pass
                continue

            if isinstance(tok, IdentifierList):
                for ident in tok.get_identifiers():
                    if isinstance(ident, Identifier):
                        _handle_identifier(ident)
                continue

            if isinstance(tok, Identifier):
                _handle_identifier(tok)
                continue

            if getattr(tok, "ttype", None) is Name:
                _maybe_add_col(tok.value)

    return cols


def _get_db_tables_and_columns(conn: sqlite3.Connection) -> Tuple[Set[str], Set[str]]:
    """
    Return (tables, columns) sets from SQLite schema; all lowercased.
    Columns are returned as a global set (unqualified).
    """
    tables = set()
    columns = set()
    for t in _list_tables(conn):
        tl = t.lower()
        if not tl:
            continue
        tables.add(tl)
        try:
            cur = conn.execute(f'PRAGMA table_info("{t}")')
            for row in cur.fetchall():
                if row and isinstance(row[1], str):
                    columns.add(row[1].lower())
        except sqlite3.Error:
            continue
    return tables, columns


def _safe_results_equal(a: List[Tuple], b: List[Tuple]) -> bool:
    # Deterministic comparison: compare exact row tuples in order.
    return a == b


@dataclass
class RewardDebugStats:
    total: int = 0
    parsed_ok: int = 0
    table_match: int = 0
    column_match: int = 0
    executed_ok: int = 0
    exact_match: int = 0


_DEBUG = RewardDebugStats()


def reset_debug_metrics() -> None:
    global _DEBUG
    _DEBUG = RewardDebugStats()


def get_debug_metrics() -> dict:
    denom = max(_DEBUG.total, 1)
    return {
        "valid_sql_rate": _DEBUG.parsed_ok / denom,
        "table_match_rate": _DEBUG.table_match / denom,
        "column_match_rate": _DEBUG.column_match / denom,
        "execution_accuracy": _DEBUG.exact_match / denom,
    }

EXECUTION_ERROR = "EXECUTION_ERROR"


def execute_sql(conn: sqlite3.Connection, sql: str, *, max_rows: int = 1000) -> Union[List[Tuple], str]:
    """
    Execute SQL safely.

    If sqlite raises ANY exception, return EXECUTION_ERROR (NOT empty list).
    """
    try:
        _with_timeout(conn, timeout_s=1.0)
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows)
        return [tuple(r) for r in rows]
    except Exception:
        return EXECUTION_ERROR


def _sqlparse_valid_select(sql: str) -> bool:
    """
    Parse validation using sqlparse:
      - parse() non-empty
      - contains a SELECT statement
    """
    if sqlparse is None:
        return False
    try:
        stmts = sqlparse.parse(sql)
        if not stmts:
            return False
        for st in stmts:
            try:
                if hasattr(st, "get_type") and st.get_type() == "SELECT":
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False

def execution_reward(pred_sql: str, db_path: str, gold_sql: str) -> float:
    try:
        sql = _normalize_sql(pred_sql)
        gold = _normalize_sql(gold_sql)

        if not sql or "SELECT" not in sql.upper():
            return -1.0

        if not _sqlparse_valid_select(sql):
            return -1.0

        reward = -0.2  # valid SQL baseline

        pred_tables = extract_tables(sql)
        gold_tables = extract_tables(gold)

        if pred_tables == gold_tables and len(gold_tables) > 0:
            reward += 0.3

        pred_cols = extract_columns(sql)
        gold_cols = extract_columns(gold)

        if gold_cols:
            overlap = len(pred_cols & gold_cols) / len(gold_cols)
            reward += 0.3 * overlap

        with _connect_readonly(db_path) as conn:
            pred_res = execute_sql(conn, sql)
            if pred_res != EXECUTION_ERROR:
                reward += 0.2

            gold_res = execute_sql(conn, gold)
            if pred_res != EXECUTION_ERROR and _safe_results_equal(pred_res, gold_res):
                return 1.0

        return max(-1.0, min(1.0, reward))

    except Exception:
        return -1.0
