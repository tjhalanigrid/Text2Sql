# from __future__ import annotations

# import json
# import subprocess
# import sys
# import argparse
# import sqlite3
# import random
# from pathlib import Path

# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from peft import PeftModel

# from prompting import encode_prompt


# def _parse_exec_accuracy(stdout: str) -> float | None:
#     for line in stdout.splitlines():
#         if line.strip().startswith("execution"):
#             try:
#                 return float(line.split()[-1])
#             except:
#                 return None
#     return None


# def main():

#     # ---------------- ARGUMENTS ----------------
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--adapter", type=str, default="checkpoints/sft_adapter_codet5")
#     parser.add_argument("--num_samples", type=int, default=1000)
#     parser.add_argument("--shuffle_dev", action="store_true")
#     parser.add_argument("--shuffle_seed", type=int, default=42)
#     parser.add_argument("--accuracy_log", type=str, default="")
#     args = parser.parse_args()

#     project_root = Path(__file__).resolve().parents[1]
#     adapter_dir = project_root / args.adapter

#     db_root = project_root / "data" / "database"
#     table_json = project_root / "data" / "tables.json"
#     dev_json = project_root / "data" / "dev.json"
#     gold_sql = project_root / "data" / "dev_gold.sql"
#     pred_path = project_root / "predictions.txt"

#     if not adapter_dir.exists():
#         raise FileNotFoundError(f"Missing adapter dir: {adapter_dir}")

#     # ---------------- DEVICE ----------------
#     device = "mps" if torch.backends.mps.is_available() else (
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )
#     print("Using device:", device)

#     # ---------------- LOAD MODEL ----------------
#     BASE_MODEL = "Salesforce/codet5-base"

#     tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)
#     model = PeftModel.from_pretrained(base, str(adapter_dir)).to(device)

#     model = model.merge_and_unload()
#     model.eval()

#     # ---------------- LOAD DATA ----------------
#     with dev_json.open() as f:
#         dev = json.load(f)

#     if args.shuffle_dev:
#         rng = random.Random(args.shuffle_seed)
#         rng.shuffle(dev)

#     dev = dev[: args.num_samples]

#     # ---------------- GENERATION CONFIG ----------------
#     gen_kwargs = dict(
#         max_new_tokens=160,
#         num_beams=4,
#         do_sample=False,
#         early_stopping=True,
#         pad_token_id=tokenizer.pad_token_id,
#         eos_token_id=tokenizer.eos_token_id,
#     )

#     print("Generating predictions...\n")

#     correct = 0
#     total = len(dev)
#     accuracy_log_fh = None

#     if args.accuracy_log:
#         accuracy_log_path = (project_root / args.accuracy_log).resolve()
#         accuracy_log_path.parent.mkdir(parents=True, exist_ok=True)
#         accuracy_log_fh = accuracy_log_path.open("w")
#         print(f"Writing running accuracy log to: {accuracy_log_path}")

#     with pred_path.open("w") as out_f, torch.no_grad():

#         for i, ex in enumerate(dev, start=1):

#             db_id = ex["db_id"]
#             question = ex["question"]
#             gold_query = ex["query"]

#             input_ids = encode_prompt(
#                 tokenizer,
#                 question,
#                 db_id,
#                 device=device,
#                 max_input_tokens=512,
#             )

#             input_ids = input_ids.unsqueeze(0).to(device)
#             attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

#             outputs = model.generate(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 **gen_kwargs
#             )

#             pred_sql = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#             out_f.write(f"{pred_sql}\t{db_id}\n")

#             # ---------------- LIVE EXECUTION CHECK ----------------
#             try:
#                 db_path = db_root / db_id / f"{db_id}.sqlite"

#                 conn = sqlite3.connect(db_path)
#                 cursor = conn.cursor()

#                 cursor.execute(pred_sql)
#                 pred_rows = cursor.fetchall()

#                 cursor.execute(gold_query)
#                 gold_rows = cursor.fetchall()

#                 conn.close()

#                 if sorted(pred_rows) == sorted(gold_rows):
#                     correct += 1

#             except Exception:
#                 pass  # execution failed

#             # 🔥 PRINT EVERY 10
#             if i % 10 == 0 or i == total:
#                 current_acc = correct / i
#                 line = f"{i}/{total} | Acc: {current_acc:.3f}"
#                 print(line)
#                 if accuracy_log_fh is not None:
#                     accuracy_log_fh.write(line + "\n")

#     if accuracy_log_fh is not None:
#         accuracy_log_fh.close()

#     print("\nGeneration finished.\n")

#     # ---------------- OFFICIAL SPIDER EVAL ----------------
#     eval_script = project_root / "spider_eval" / "evaluation.py"

#     cmd = [
#         sys.executable,
#         str(eval_script),
#         "--gold", str(gold_sql),
#         "--pred", str(pred_path),
#         "--etype", "exec",
#         "--db", str(db_root),
#         "--table", str(table_json),
#     ]

#     print("Running Spider evaluation...")
#     proc = subprocess.run(cmd, capture_output=True, text=True)

#     print(proc.stdout)

#     exec_acc = _parse_exec_accuracy(proc.stdout)
#     if exec_acc is not None:
#         print(f"\n🎯 Official Execution Accuracy: {exec_acc*100:.2f}%")
#     else:
#         print("Could not parse accuracy.")


# if __name__ == "__main__":
#     main()

import json
import subprocess
import sys
import argparse
import random
import sqlite3
import time
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Assuming you have a prompting.py that has encode_prompt
from prompting import encode_prompt

# -------------------------------
# LIVE CHECK HELPERS
# -------------------------------
def normalize_sql(sql):
    """Basic normalization for the live progress bar."""
    sql = sql.replace('"', "'")
    sql = re.sub(r"\s+", " ", sql)
    return sql.strip().lower().rstrip(";")

def check_execution(pred_sql, gold_sql, db_path):
    """Basic execution check for the live progress bar."""
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        
        # 2-second timeout so the live tracker doesn't freeze forever
        start_time = time.monotonic()
        def timeout_handler():
            return 1 if (time.monotonic() - start_time) > 2.0 else 0
        conn.set_progress_handler(timeout_handler, 10000)

        cursor = conn.cursor()
        cursor.execute(pred_sql)
        pred_res = cursor.fetchall()
        
        cursor.execute(gold_sql)
        gold_res = cursor.fetchall()
        conn.close()
        
        # Simple sorted check for the live tracker
        return sorted(pred_res) == sorted(gold_res)
    except Exception:
        return False

# -------------------------------
# SPIDER PARSER
# -------------------------------
def _parse_spider_accuracy(stdout: str, metric_type: str) -> float | None:
    for line in stdout.splitlines():
        if metric_type == "exec" and line.strip().startswith("execution"):
            try: return float(line.split()[-1])
            except: pass
        elif metric_type == "match" and line.strip().startswith("exact"):
            try: return float(line.split()[-1])
            except: pass
    return None

# -------------------------------
# MAIN
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True, help="Path to your SFT or RLHF checkpoint")
    parser.add_argument("--num_samples", type=int, default=1034, help="Number of samples to evaluate")
    parser.add_argument("--shuffle_dev", action="store_true")
    parser.add_argument("--shuffle_seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    adapter_dir = project_root / args.adapter

    db_root = project_root / "data" / "database"
    table_json = project_root / "data" / "tables.json"
    dev_json = project_root / "data" / "dev.json"
    
    pred_path = project_root / "temp_predictions.txt"
    temp_gold_path = project_root / "temp_gold.sql"

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter dir: {adapter_dir}")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    BASE_MODEL = "Salesforce/codet5-base"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Model: {args.adapter}...")
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).to(device)
    model = model.merge_and_unload()
    model.eval()

    with dev_json.open() as f:
        dev = json.load(f)

    if args.shuffle_dev:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(dev)

    dev = dev[: args.num_samples]
    total = len(dev)

    gen_kwargs = dict(
        max_new_tokens=160,
        num_beams=4,
        do_sample=False,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    print(f"\n🚀 Generating and live-tracking {total} samples...\n")

    em_correct = 0
    ex_correct = 0

    with pred_path.open("w") as out_pred, temp_gold_path.open("w") as out_gold, torch.no_grad():
        for i, ex in enumerate(dev, start=1):
            db_id = ex["db_id"]
            question = ex["question"]
            gold_query = ex["query"]
            db_path = db_root / db_id / f"{db_id}.sqlite"

            # Generate
            input_ids = encode_prompt(tokenizer, question, db_id, device=device, max_input_tokens=512)
            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
            pred_sql = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Write to files for official spider eval later
            out_pred.write(f"{pred_sql}\n")
            out_gold.write(f"{gold_query}\t{db_id}\n")

            # --- LIVE TRACKING CHECKS ---
            if normalize_sql(pred_sql) == normalize_sql(gold_query):
                em_correct += 1
            if check_execution(pred_sql, gold_query, db_path):
                ex_correct += 1

            # Print progress every 50 loops
            if i % 50 == 0 or i == total:
                print(f"Progress: {i}/{total} | Current EM: {(em_correct/i)*100:.2f}% | Current EX: {(ex_correct/i)*100:.2f}%")

    print("\nGeneration finished. Running Official Spider Evaluations for final numbers...\n")

    eval_script = project_root / "spider_eval" / "evaluation.py"

    # 1. RUN EXACT MATCH EVAL
    cmd_match = [
        sys.executable, str(eval_script),
        "--gold", str(temp_gold_path),
        "--pred", str(pred_path),
        "--etype", "match",
        "--db", str(db_root),
        "--table", str(table_json),
    ]
    proc_match = subprocess.run(cmd_match, capture_output=True, text=True)
    exact_acc = _parse_spider_accuracy(proc_match.stdout, "match")

    # 2. RUN EXECUTION EVAL
    cmd_exec = [
        sys.executable, str(eval_script),
        "--gold", str(temp_gold_path),
        "--pred", str(pred_path),
        "--etype", "exec",
        "--db", str(db_root),
        "--table", str(table_json),
    ]
    proc_exec = subprocess.run(cmd_exec, capture_output=True, text=True)
    exec_acc = _parse_spider_accuracy(proc_exec.stdout, "exec")

    print("==========================================")
    print(f"🎯 OFFICIAL SPIDER RESULTS FOR: {args.adapter}")
    print("==========================================")
    
    if exact_acc is not None:
        print(f"Exact Set Match Accuracy  : {exact_acc*100:.2f}%")
    else:
        print("Exact Set Match Accuracy  : Could not parse output")
        
    if exec_acc is not None:
        print(f"Execution Accuracy        : {exec_acc*100:.2f}%")
    else:
        print("Execution Accuracy        : Could not parse output")
    print("==========================================\n")

if __name__ == "__main__":
    main()