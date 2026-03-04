
import json
import sqlite3
import argparse
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ---------------- PROMPT (IDENTICAL TO TRAINING) ----------------
def build_prompt(question, schema):
    return f"""
Database Schema:
{schema}

Translate English to SQL:
{question}
SQL:
"""

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


# ---------------- EXECUTION CHECK WITH TIMEOUT ----------------
def execution_match(pred_sql, gold_sql, db_path):
    try:
        conn = sqlite3.connect(db_path)
        
        # --- 5-SECOND TIMEOUT SO EVALUATION DOESN'T FREEZE ---
        start_time = time.monotonic()
        def timeout_handler():
            return 1 if (time.monotonic() - start_time) > 5.0 else 0
        conn.set_progress_handler(timeout_handler, 10000)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1034)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    dev_json = project_root / "data" / "dev.json"
    db_root = project_root / "data" / "database"

    # 🎯 Added CUDA support for Nvidia GPUs
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    base_model = "facebook/bart-base"
    print(f"Loading Base: {base_model}")
    print(f"Loading Adapter: {args.adapter}")
     
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model).to(device)
    model = PeftModel.from_pretrained(base, args.adapter).to(device)
    model = model.merge_and_unload()

    with open(dev_json) as f:
        dev = json.load(f)[: args.num_samples]

    correct = 0

    print(f"Evaluating {len(dev)} examples...\n")

    for i, ex in enumerate(dev, 1):
        question = ex["question"]
        db_id = ex["db_id"]
        gold_sql = ex["query"]

        db_path = db_root / db_id / f"{db_id}.sqlite"
        schema = load_schema(db_path)

        prompt = build_prompt(question, schema)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                num_beams=4,
            )

        pred_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "SQL:" in pred_sql:
            pred_sql = pred_sql.split("SQL:")[-1].strip()

        match = execution_match(pred_sql, gold_sql, db_path)

        if match:
            correct += 1

        if i % 10 == 0:
            print(f"{i}/{len(dev)} | Acc: {correct/i:.3f}")

    print("\n=============================")
    print(f"FINAL EXECUTION ACCURACY: {correct/len(dev)*100:.2f}%")
    print("=============================")


if __name__ == "__main__":
    main()