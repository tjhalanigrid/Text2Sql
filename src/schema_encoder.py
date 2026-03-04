import sqlite3


class SchemaEncoder:

    def __init__(self, db_root):
        self.db_root = db_root

    def get_tables_and_columns(self, db_id):
        db_path = self.db_root / db_id / f"{db_id}.sqlite"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()

        schema = {}

        for (table,) in tables:
            cols = cursor.execute(f"PRAGMA table_info({table});").fetchall()
            col_names = [c[1] for c in cols]
            schema[table] = col_names

        conn.close()
        return schema

    # -----------------------------------
    # Strategy 1: Structured (current)
    # -----------------------------------
    def structured_schema(self, db_id):
        schema = self.get_tables_and_columns(db_id)

        lines = []
        for table, cols in schema.items():
            lines.append(f"{table}({', '.join(cols)})")

        return "\n".join(lines)

    # -----------------------------------
    # Strategy 2: Natural Language
    # -----------------------------------
    def natural_language_schema(self, db_id):
        schema = self.get_tables_and_columns(db_id)

        lines = []
        for table, cols in schema.items():
            col_text = ", ".join(cols)
            lines.append(f"The table '{table}' contains the columns: {col_text}.")

        return "\n".join(lines)
