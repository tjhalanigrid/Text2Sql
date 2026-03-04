import sqlite3

def get_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    schema_text = ""

    for table in tables:
        table = table[0]
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        col_names = [col[1] for col in columns]

        schema_text += f"{table}({', '.join(col_names)})\n"

    conn.close()
    return schema_text
