import json
import sqlite3
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- PROMPT (same style as training) ----------------
def build_prompt(question, schema):
    return f"""translate English to SQL:

Schema:
{schema}

Question:
{question}

SQL:"""

# ---------------- LOAD SCHEMA ----------------
def load_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    tables = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    ).fetchall()

    schema = ""
    for (table,) in tables:
        cols = cursor.execute(f"PRAGMA table_info({table});").fetchall()
        col_names = [c[1] for c in cols]
        schema += f"{table}({', '.join(col_names)})\n"

    conn.close()
    return schema

# ---------------- EXECUTION MATCH ----------------
def execution_match(pred_sql, gold_sql, db_path):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute(pred_sql)
        pred = cur.fetchall()

        cur.execute(gold_sql)
        gold = cur.fetchall()

        conn.close()
        return pred == gold

    except Exception:
        return False

# ---------------- MAIN ----------------
def main():
    project_root = Path(__file__).resolve().parents[1]

    dev_json = project_root / "data" / "dev.json"
    db_root = project_root / "data" / "database"

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("Loading BASE CodeT5...")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base").to(device)
    model.eval()

    with open(dev_json) as f:
        dev = json.load(f)[:100]

    correct = 0

    print(f"\nEvaluating {len(dev)} samples...\n")

    for i, ex in enumerate(dev, 1):
        question = ex["question"]
        db_id = ex["db_id"]
        gold_sql = ex["query"]

        db_path = db_root / db_id / f"{db_id}.sqlite"
        schema = load_schema(db_path)

        prompt = build_prompt(question, schema)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                num_beams=4,
                do_sample=False
            )

        pred_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "SQL:" in pred_sql:
            pred_sql = pred_sql.split("SQL:")[-1].strip()

        if execution_match(pred_sql, gold_sql, db_path):
            correct += 1

        if i % 10 == 0:
            print(f"{i}/100 | Accuracy: {correct/i:.3f}")

    print("\n=============================")
    print(f"BASE MODEL ACCURACY: {correct}% / 100 = {correct}%")
    print("=============================")

if __name__ == "__main__":
    main()
