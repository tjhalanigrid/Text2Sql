import json
import subprocess
import sys
import argparse
import random
import sqlite3
import time
import re
import matplotlib.pyplot as plt
import numpy as np
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
    sql = sql.replace('"', "'")
    sql = re.sub(r"\s+", " ", sql)
    return sql.strip().lower().rstrip(";")

def check_execution(pred_sql, gold_sql, db_path):
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        
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
    parser.add_argument("--adapter", type=str, required=True, help="Path to your checkpoint")
    parser.add_argument("--base_model", type=str, required=True, help="E.g., facebook/bart-base, t5-small")
    parser.add_argument("--model_name", type=str, required=True, help="Name for the plot label (e.g., 'BART RLHF')")
    parser.add_argument("--num_samples", type=int, default=700)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    adapter_dir = project_root / args.adapter

    db_root = project_root / "data" / "database"
    table_json = project_root / "data" / "tables.json"
    dev_json = project_root / "data" / "dev.json"
    
    pred_path = project_root / "temp_predictions.txt"
    temp_gold_path = project_root / "temp_gold.sql"
    
    # NEW: Plot directory setup
    plot_dir = project_root / "comparison_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    results_json_path = plot_dir / "all_metrics.json"

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter dir: {adapter_dir}")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Base Model: {args.base_model} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSeq2SeqLM.from_pretrained(args.base_model).to(device)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).to(device)
    model = model.merge_and_unload()
    model.eval()

    with dev_json.open() as f:
        dev = json.load(f)[: args.num_samples]
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
            
            out_pred.write(f"{pred_sql}\n")
            out_gold.write(f"{gold_query}\t{db_id}\n")

            # --- PRINT FIRST 3 EXAMPLES ---
            if i <= 3:
                print(f"--- 🔍 Example {i} ---")
                print(f"Q   : {question}")
                print(f"Gold: {gold_query}")
                print(f"Pred: {pred_sql}")
                print("-" * 25)

            # --- LIVE TRACKING CHECKS ---
            if normalize_sql(pred_sql) == normalize_sql(gold_query):
                em_correct += 1
            if check_execution(pred_sql, gold_query, db_path):
                ex_correct += 1

            if i % 50 == 0 or i == total:
                print(f"Progress: {i}/{total} | Current EM: {(em_correct/i)*100:.2f}% | Current EX: {(ex_correct/i)*100:.2f}%")

    print("\nRunning Official Spider Evaluations...")
    eval_script = project_root / "spider_eval" / "evaluation.py"

    proc_match = subprocess.run([sys.executable, str(eval_script), "--gold", str(temp_gold_path), "--pred", str(pred_path), "--etype", "match", "--db", str(db_root), "--table", str(table_json)], capture_output=True, text=True)
    exact_acc = _parse_spider_accuracy(proc_match.stdout, "match")

    proc_exec = subprocess.run([sys.executable, str(eval_script), "--gold", str(temp_gold_path), "--pred", str(pred_path), "--etype", "exec", "--db", str(db_root), "--table", str(table_json)], capture_output=True, text=True)
    exec_acc = _parse_spider_accuracy(proc_exec.stdout, "exec")

    print("\n==========================================")
    print(f"🎯 RESULTS FOR: {args.model_name}")
    print("==========================================")
    exact_val = exact_acc * 100 if exact_acc else 0
    exec_val = exec_acc * 100 if exec_acc else 0
    print(f"Exact Match : {exact_val:.2f}%")
    print(f"Execution   : {exec_val:.2f}%")
    print("==========================================\n")

    # -------------------------------
    # SAVE JSON & GENERATE PLOT
    # -------------------------------
    if results_json_path.exists():
        with open(results_json_path, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results[args.model_name] = {"EM": exact_val, "EX": exec_val}

    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    labels = list(all_results.keys())
    em_vals = [all_results[k]["EM"] for k in labels]
    ex_vals = [all_results[k]["EX"] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(max(8, len(labels) * 1.5), 6))
    plt.bar(x - width/2, em_vals, width, label='Exact Match', color='#3498db')
    plt.bar(x + width/2, ex_vals, width, label='Execution', color='#2ecc71')

    plt.ylabel('Accuracy (%)', fontweight='bold')
    plt.title('Model Comparison: Exact Match vs Execution Accuracy', fontweight='bold', fontsize=14)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.legend()
    plt.ylim(0, max(max(em_vals, default=0), max(ex_vals, default=0)) + 15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Attach labels to bars
    for i in range(len(labels)):
        plt.text(x[i] - width/2, em_vals[i] + 1, f"{em_vals[i]:.1f}%", ha='center', fontsize=9)
        plt.text(x[i] + width/2, ex_vals[i] + 1, f"{ex_vals[i]:.1f}%", ha='center', fontsize=9)

    plt.tight_layout()
    plot_path = plot_dir / "accuracy_comparison.png"
    plt.savefig(plot_path, dpi=300)
    print(f"📈 Updated comparison plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
