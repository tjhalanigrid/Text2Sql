import json
import sqlite3
import torch
import re
import time
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = PROJECT_ROOT / "data" / "database"

# -------------------------------
# 1. NORMALIZATION FOR EXACT MATCH
# -------------------------------
def normalize_sql(sql):
    """Cleans SQL to make Exact Match grading fair (ignores spacing/cases)."""
    sql = sql.replace('"', "'")        # Standardize quotes
    sql = re.sub(r"\s+", " ", sql)     # Remove extra spaces/newlines
    sql = sql.strip().lower()          # Lowercase everything
    sql = sql.rstrip(";")              # Remove trailing semicolons
    return sql

# -------------------------------
# 2. EXECUTION ACCURACY CHECK
# -------------------------------
def check_execution(pred_sql, gold_sql, db_path):
    """Runs both queries and checks if the output rows/columns match."""
    try:
        conn = sqlite3.connect(db_path)
        # Handle bad characters in Spider DBs
        conn.text_factory = lambda b: b.decode(errors='ignore')
        
        # 5-second timeout
        start_time = time.monotonic()
        def timeout_handler():
            return 1 if (time.monotonic() - start_time) > 5.0 else 0
        conn.set_progress_handler(timeout_handler, 10000)

        cursor = conn.cursor()

        # Get Predicted Result
        cursor.execute(pred_sql)
        pred_res = cursor.fetchall()

        # Get Gold Result
        cursor.execute(gold_sql)
        gold_res = cursor.fetchall()

        conn.close()
        return pred_res == gold_res
    except Exception:
        return False

# -------------------------------
# 3. LOAD SCHEMA
# -------------------------------
def load_schema(db_path):
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode(errors='ignore')
    cursor = conn.cursor()
    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    schema = ""
    for (table,) in tables:
        cols = cursor.execute(f"PRAGMA table_info({table});").fetchall()
        col_names = [c[1] for c in cols]
        schema += f"{table}({', '.join(col_names)})\n"
    conn.close()
    return schema

# -------------------------------
# 4. MAIN PIPELINE
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True, help="Path to your SFT or RLHF checkpoint")
    parser.add_argument("--num_samples", type=int, default=1034, help="How many samples to evaluate")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    base_model = "Salesforce/codet5-base"

    print(f"\n🚀 Loading Model from: {args.adapter}")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model).to(device)
    model = PeftModel.from_pretrained(base, args.adapter).to(device)
    model = model.merge_and_unload()
    model.eval()

    dev_json = PROJECT_ROOT / "data" / "dev.json"
    with open(dev_json) as f:
        dev = json.load(f)[:args.num_samples]

    em_correct = 0
    ex_correct = 0
    total = len(dev)

    print(f"\n📊 Evaluating {total} queries for BOTH Exact Match and Execution Accuracy...\n")

    for i, ex in enumerate(dev, 1):
        question = ex["question"]
        gold_sql = ex["query"]
        db_id = ex["db_id"]
        db_path = DB_ROOT / db_id / f"{db_id}.sqlite"

        # Generate SQL
        schema = load_schema(db_path)
        prompt = f"Database Schema:\n{schema}\nTranslate English to SQL:\n{question}\nSQL:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, num_beams=4, do_sample=False)
        
        pred_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "SQL:" in pred_sql:
            pred_sql = pred_sql.split("SQL:")[-1].strip()

        # --- METRIC 1: EXACT MATCH ---
        is_em = (normalize_sql(pred_sql) == normalize_sql(gold_sql))
        if is_em:
            em_correct += 1

        # --- METRIC 2: EXECUTION ACCURACY ---
        is_ex = check_execution(pred_sql, gold_sql, db_path)
        if is_ex:
            ex_correct += 1

        if i % 50 == 0 or i == total:
            print(f"Progress: {i}/{total} | Current EM: {(em_correct/i)*100:.2f}% | Current EX: {(ex_correct/i)*100:.2f}%")

    # Final Results
    final_em = (em_correct / total) * 100
    final_ex = (ex_correct / total) * 100

    print("\n==========================================")
    print(f"🎯 FINAL RESULTS FOR: {args.adapter}")
    print("==========================================")
    print(f"Exact Match (EM) Accuracy      : {final_em:.2f}%")
    print(f"Execution (EX) Accuracy        : {final_ex:.2f}%")
    print("==========================================\n")

if __name__ == "__main__":
    main()
