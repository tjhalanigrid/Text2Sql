import re

def extract_tables(schema_text):
    """
    Convert schema string into table list
    genre(a,b,c) -> genre
    """
    tables = []
    for line in schema_text.split("\n"):
        line = line.strip()
        if "(" in line:
            table = line.split("(")[0].strip()
            tables.append(table.lower())
    return set(tables)


def clean_sql(sql, schema, question):

    sql = sql.strip()

    # remove duplicate SELECT
    sql = re.sub(r'(SELECT\s+)+', 'SELECT ', sql, flags=re.IGNORECASE)

    # remove hallucinated function calls: song(song_name) -> song_name
    sql = re.sub(r'(\w+)\((\w+)\)', r'\2', sql)

    # remove incomplete GROUP BY
    sql = re.sub(r'GROUP BY\s*$', '', sql, flags=re.IGNORECASE)

    # add semicolon
    if not sql.endswith(";"):
        sql += ";"

    # -------- TABLE VALIDATION --------
    valid_tables = extract_tables(schema)

    def fix_table(match):
        table = match.group(1).lower()

        if table in valid_tables:
            return f"FROM {table}"

        return match.group(0)

    sql = re.sub(r'FROM\s+(\w+)', fix_table, sql, flags=re.IGNORECASE)

    return sql