import json
import sqlite3
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = PROJECT_ROOT / "data" / "database"

# Added CUDA fallback for consistency
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ================= LOAD MODEL =================
def load_model(adapter_path):
    base_name = "Salesforce/codet5-base"

    # 🐛 FIXED: Convert relative path to absolute path to prevent Hugging Face 404 errors
    abs_path = (PROJECT_ROOT / adapter_path).resolve()
    if not abs_path.exists():
        raise FileNotFoundError(f"Adapter not found at: {abs_path}")

    print(f"\nLoading model from: {abs_path}")

    # 🐛 FIXED: Added fallback in case tokenizer isn't saved in the adapter folder
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(abs_path), local_files_only=True)
    except Exception:
        print("Adapter tokenizer missing — using base tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(base_name)

    base = AutoModelForSeq2SeqLM.from_pretrained(base_name).to(DEVICE)
    model = PeftModel.from_pretrained(base, str(abs_path)).to(DEVICE)
    model.eval()

    return tokenizer, model


# ================= SCHEMA =================
def load_schema(db_id):
    db_path = DB_ROOT / db_id / f"{db_id}.sqlite"
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


# ================= GENERATE =================
def generate_sql(tokenizer, model, question, db_id):
    schema = load_schema(db_id)

    prompt = f"""
Database Schema:
{schema}

Translate English to SQL:
{question}
SQL:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            num_beams=4,
            do_sample=False
        )

    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "SQL:" in sql:
        sql = sql.split("SQL:")[-1]

    return sql.strip()


# ================= EXECUTE =================
def try_execute(sql, db_id):
    db_path = DB_ROOT / db_id / f"{db_id}.sqlite"
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        cur.fetchall()
        conn.close()
        return True
    except:
        return False


# ================= MAIN =================
def main():
    # paths (change if needed)
    SFT_MODEL = "checkpoints/sft_adapter_codet5" # Ensure this matches your actual SFT folder name!
    RLHF_MODEL = "checkpoints/best_rlhf_model"

    tokenizer_sft, model_sft = load_model(SFT_MODEL)
    tokenizer_rl, model_rl = load_model(RLHF_MODEL)

    human_eval_path = PROJECT_ROOT / "data/human_eval.json"
    with open(human_eval_path) as f:
        questions = json.load(f)

    sft_success = 0
    rl_success = 0

    print("\nRunning Human Evaluation...\n")

    for i, q in enumerate(questions, 1):
        db = q["db_id"]
        question = q["question"]

        sql_sft = generate_sql(tokenizer_sft, model_sft, question, db)
        sql_rl = generate_sql(tokenizer_rl, model_rl, question, db)

        ok_sft = try_execute(sql_sft, db)
        ok_rl = try_execute(sql_rl, db)

        if ok_sft:
            sft_success += 1
        if ok_rl:
            rl_success += 1

        print(f"\nQ{i}: {question}")
        print(f"SFT : {'OK' if ok_sft else 'FAIL'}")
        print(f"RLHF: {'OK' if ok_rl else 'FAIL'}")

    print("\n=============================")
    print("HUMAN EVALUATION RESULT")
    print("=============================")
    print(f"SFT  Success: {sft_success}/{len(questions)} = {sft_success/len(questions)*100:.2f}%")
    print(f"RLHF Success: {rl_success}/{len(questions)} = {rl_success/len(questions)*100:.2f}%")
    print("=============================\n")


if __name__ == "__main__":
    main()