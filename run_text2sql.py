from sql_cleaner import clean_sql
from schema_utils import get_schema

import torch
import sqlite3
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ALWAYS use base tokenizer (important for T5)
BASE_MODEL = "google/flan-t5-small"

# trained weights
MODEL_PATH = "/Users/tjhalani/text2sql_project/outputs/model"

# database
DB_PATH = "/Users/tjhalani/text2sql_project/data/database/music_1/music_1.sqlite"


print("Loading Text2SQL model...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.eval()


# ----------- LOAD SCHEMA -----------
SCHEMA = get_schema(DB_PATH)

print("\nConnected to database:", DB_PATH)
print("\nDetected Database Schema:\n")
print(SCHEMA)

print("\nModel ready! Ask questions (type 'exit' to stop)\n")


# ---------- SQL GENERATION ----------
def generate_sql(question):

    prompt = f"""
Database Schema:
{SCHEMA}

Translate English to SQL:
{question}
SQL:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            num_beams=4,
            early_stopping=True
        )

    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # remove prompt echo
    if "SQL:" in sql:
        sql = sql.split("SQL:")[-1]

    return sql.strip()


# ---------- SQL EXECUTION ----------
def run_sql(sql):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        return f"SQL ERROR: {e}"


# ---------- MAIN LOOP ----------
while True:
    q = input("Ask: ")

    if q.lower() == "exit":
        break

    # model output
    dirty_sql = generate_sql(q)

    # clean + fix sql
    cleaned_sql = clean_sql(dirty_sql, SCHEMA, q)

    print("\nGenerated SQL:")
    print(dirty_sql)

    print("\nCleaned SQL:")
    print(cleaned_sql)

    # execute
    result = run_sql(cleaned_sql)

    print("\nResult:")
    print(result)
    print("\n----------------------------------")
