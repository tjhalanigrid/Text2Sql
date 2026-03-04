# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import json

# import subprocess

# import argparse
# from pathlib import Path

# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from peft import PeftModel

# # IMPORTANT: must match training prompt format
# from prompting import build_prompt
# from schema_utils import get_schema as get_db_schema


# def _parse_exec_accuracy(stdout: str):
#     for line in stdout.splitlines():
#         if line.strip().startswith("execution"):
#             parts = line.split()
#             try:
#                 return float(parts[-1])
#             except Exception:
#                 return None
#     return None


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--adapter", type=str, default="checkpoints/best_rlhf_model")
#     parser.add_argument("--num_samples", type=int, default=200)
#     args = parser.parse_args()

#     project_root = Path(__file__).resolve().parents[1]
#     adapter_dir = project_root / args.adapter

#     if not adapter_dir.exists():
#         raise FileNotFoundError(f"Adapter not found: {adapter_dir}")

#     db_root = project_root / "data" / "database"
#     table_json = project_root / "data" / "tables.json"
#     dev_json = project_root / "data" / "dev.json"
#     gold_sql = project_root / "data" / "dev_gold.sql"
#     pred_path = project_root / "predictions_rl.txt"

#     device = "mps" if torch.backends.mps.is_available() else "cpu"

#     # ---- LOAD MODEL (CodeT5 + LoRA) ----
#     base_model = "Salesforce/codet5-base"

#     tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
#     base = AutoModelForSeq2SeqLM.from_pretrained(base_model).to(device)
#     model = PeftModel.from_pretrained(base, str(adapter_dir)).to(device)

#     # merge LoRA for faster inference
#     model = model.merge_and_unload()
#     model.eval()
#     model.config.use_cache = True

#     if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
#         tokenizer.pad_token = tokenizer.eos_token

#     # ---- LOAD DATA ----
#     with dev_json.open() as f:
#         dev = json.load(f)

#     dev = dev[: args.num_samples]

#     gen_kwargs = dict(
#         max_new_tokens=120,
#         do_sample=False,
#         num_beams=1,
#         pad_token_id=tokenizer.pad_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#     )

#     print(f"Generating {len(dev)} predictions...")

#     with pred_path.open("w") as out_f, torch.no_grad():
#         for i, ex in enumerate(dev, start=1):
#             db_id = ex["db_id"]
#             question = ex["question"]

#             db_path = db_root / db_id / f"{db_id}.sqlite"
#             schema = get_db_schema(str(db_path))
#             prompt = build_prompt(question, schema, use_schema=True)

#             inputs = tokenizer(
#                 prompt,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=512
#             ).to(device)

#             out = model.generate(**inputs, **gen_kwargs)
#             pred_sql = tokenizer.decode(out[0], skip_special_tokens=True).strip()

#             out_f.write(f"{pred_sql}\t{db_id}\n")

#             if i % 20 == 0 or i == len(dev):
#                 print(f"{i}/{len(dev)} done")

#     # ---- SPIDER OFFICIAL EVAL ----
#     eval_script = project_root / "spider_eval" / "evaluation.py"

#     cmd = [
#         sys.executable,
#         str(eval_script),
#         "--gold",
#         str(gold_sql),
#         "--pred",
#         str(pred_path),
#         "--etype",
#         "exec",
#         "--db",
#         str(db_root),
#         "--table",
#         str(table_json),
#     ]

#     print("\nRunning Spider execution evaluation...\n")
#     proc = subprocess.run(cmd, capture_output=True, text=True)

#     if proc.returncode != 0:
#         print(proc.stdout)
#         print(proc.stderr)
#         sys.exit(proc.returncode)

#     print(proc.stdout)

#     acc = _parse_exec_accuracy(proc.stdout)
#     if acc is not None:
#         print(f"\nFINAL EXECUTION ACCURACY: {acc*100:.2f}%")
#     else:
#         print("Could not parse execution accuracy")


# if __name__ == "__main__":
#     main()


import json
import sqlite3
import argparse
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ---------------- PROMPT (FIXED TO PERFECTLY MATCH RLHF TRAINING) ----------------
def build_prompt(question, schema):
    return f"translate English to SQL:\n\nSchema:\n{schema}\n\nQuestion:\n{question}\n\nSQL:"

# ---------------- LOAD SCHEMA (FIXED TO MATCH TRAINING FORMAT) ----------------
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
        # Space-separated, not newline-separated, just like the RLHF script
        schema += f"{table}({', '.join(col_names)}) " 

    conn.close()
    return schema.strip()


# ---------------- EXECUTION CHECK WITH TIMEOUT ----------------
def execution_match(pred_sql, gold_sql, db_path):
    try:
        conn = sqlite3.connect(db_path)
        
        # --- 5-SECOND TIMEOUT SO THE SCRIPT DOESN'T HANG ---
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
    # 🎯 Set the default directly to your best RLHF model!
    parser.add_argument("--adapter", type=str, default="checkpoints/rlhf_t5_best")
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    
    # Resolve adapter path safely
    adapter_path = project_root / args.adapter

    dev_json = project_root / "data" / "dev.json"
    db_root = project_root / "data" / "database"

    # 🎯 Added CUDA support
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    base_model = "t5-small"
    print(f"Loading Base: {base_model}")
    print(f"Loading Adapter: {adapter_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model).to(device)
    model = PeftModel.from_pretrained(base, str(adapter_path)).to(device)
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