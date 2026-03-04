import argparse
import json
import os
import sqlite3
from contextlib import closing


def execute_sql(db_path: str, sql: str):
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
        return {"ok": True, "rows": rows, "error": None}
    except Exception as e:
        return {"ok": False, "rows": [], "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Safely execute SQL against a Spider SQLite DB.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--db_path", type=str, help="Path to SQLite database file")
    group.add_argument("--db_id", type=str, help="Spider database id (uses data/database/<db_id>/<db_id>.sqlite)")
    parser.add_argument("--sql", type=str, required=True, help="SQL to execute")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.db_path:
        db_path = args.db_path
    else:
        db_path = os.path.join(project_root, "data", "database", args.db_id, f"{args.db_id}.sqlite")

    result = execute_sql(db_path, args.sql)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

