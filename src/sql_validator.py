import sqlite3
import re
from pathlib import Path

class SQLValidator:

    def __init__(self, db_root):
        self.db_root = Path(db_root)

    # ---------------------------
    # Load schema
    # ---------------------------
    def load_schema(self, db_id):
        db_path = self.db_root / db_id / f"{db_id}.sqlite"

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()

        schema = {}

        for (table,) in tables:
            cols = cursor.execute(f"PRAGMA table_info({table});").fetchall()
            schema[table.lower()] = [c[1].lower() for c in cols]

        conn.close()
        return schema


    # ---------------------------
    # Basic syntax check
    # ---------------------------
    def basic_structure_valid(self, sql):
        s = sql.lower()

        if "select" not in s or "from" not in s:
            return False, "Missing SELECT or FROM"

        if len(s.split()) < 4:
            return False, "Too short to be SQL"

        return True, None


    # ---------------------------
    # Extract identifiers
    # ---------------------------
    def extract_identifiers(self, sql):
        tokens = re.findall(r"[A-Za-z_]+", sql.lower())
        return set(tokens)


    # ---------------------------
    # Table validation
    # ---------------------------
    def validate_tables(self, sql, schema):
        words = self.extract_identifiers(sql)
        tables = set(schema.keys())

        used_tables = [w for w in words if w in tables]

        if not used_tables:
            return False, "No valid table used"

        return True, None


    # ---------------------------
    # Column validation
    # ---------------------------
    def validate_columns(self, sql, schema):
        words = self.extract_identifiers(sql)

        valid_columns = set()
        for cols in schema.values():
            valid_columns.update(cols)

        # ignore SQL keywords
        keywords = {
            "select","from","where","join","on","group","by",
            "order","limit","count","sum","avg","min","max",
            "and","or","in","like","distinct","asc","desc"
        }

        invalid = []
        for w in words:
            if w not in valid_columns and w not in schema and w not in keywords:
                if not w.isdigit():
                    invalid.append(w)

        # allow small hallucinations but block many
        if len(invalid) > 3:
            return False, f"Too many unknown identifiers: {invalid[:5]}"

        return True, None


    # ---------------------------
    # Dangerous query protection
    # ---------------------------
    def block_dangerous(self, sql):
        bad = ["drop", "delete", "update", "insert", "alter"]

        s = sql.lower()
        for b in bad:
            if b in s:
                return False, f"Dangerous keyword detected: {b}"

        return True, None


    # ---------------------------
    # Main validation
    # ---------------------------
    def validate(self, sql, db_id):

        schema = self.load_schema(db_id)

        checks = [
            self.block_dangerous(sql),
            self.basic_structure_valid(sql),
            self.validate_tables(sql, schema),
            self.validate_columns(sql, schema),
        ]

        for ok, msg in checks:
            if not ok:
                return False, msg

        return True, None
