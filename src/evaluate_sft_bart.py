from __future__ import annotations

import json
import subprocess
import sys
import argparse
import re
import sqlite3
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from prompting import encode_prompt


# ---------------- SQL CLEAN ----------------
def extract_sql(text: str) -> str:
    text = text.strip()

    if "SQL:" in text:
        text = text.split("SQL:")[-1]

    match = re.search(r"(SELECT .*?)(?:$)", text, re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1)

    text = text.replace('"', "'")
    text = re.sub(r"\s+", " ", text).strip()

    if not text.endswith(";"):
        text += ";"

    return text


# ---------------- ROBUST ACC PARSER ----------------
def parse_exec_accuracy(stdout: str):
    for line in stdout.splitlines():
        if "execution" in line.lower():
            numbers = re.findall(r"\d+\.\d+", line)
            if numbers:
                return float(numbers[-1])
    return None


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default="checkpoints/sft_best_bart_2")
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    adapter_dir = project_root / args.adapter

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_dir}")

    db_root = project_root / "data/database"
    table_json = project_root / "data/tables.json"
    dev_json = project_root / "data/dev.json"
    gold_sql_file = project_root / "data/dev_gold.sql"
    pred_sql_file = project_root / "pred.sql"

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    # -------- LOAD MODEL --------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    BASE_MODEL = "facebook/bart-base"
    print(f"Loading base model {BASE_MODEL}...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)
    model = model.merge_and_unload()
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------- LOAD DATA --------
    with open(dev_json) as f:
        dev = json.load(f)[: args.num_samples]

    print("Generating SQL predictions...\n")

    correct = 0
    total = len(dev)

    with open(pred_sql_file, "w") as f, torch.no_grad():

        for i, ex in enumerate(dev, 1):

            question = ex["question"]
            db_id = ex["db_id"]
            gold_query = ex["query"]

            prompt_ids = encode_prompt(
                tokenizer,
                question,
                db_id,
                device=device,
                max_input_tokens=512,
            )

            input_ids = prompt_ids.unsqueeze(0).to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=160,
                num_beams=4,
                do_sample=False,
            )

            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred_sql = extract_sql(pred)

            f.write(f"{pred_sql}\t{db_id}\n")

            # -------- LIVE EXECUTION CHECK --------
            try:
                db_path = db_root / db_id / f"{db_id}.sqlite"

                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                cursor.execute(pred_sql)
                pred_rows = cursor.fetchall()

                cursor.execute(gold_query)
                gold_rows = cursor.fetchall()

                conn.close()

                # order insensitive comparison
                if sorted(pred_rows) == sorted(gold_rows):
                    correct += 1

            except Exception:
                pass  # execution failed

            if i % 10 == 0 or i == total:
                current_acc = correct / i
                print(f"{i}/{total} | Acc: {current_acc:.3f}")

    print("\nGeneration finished.\n")

    # -------- RUN OFFICIAL SPIDER EVAL --------
    eval_script = project_root / "spider_eval/evaluation.py"
    if (project_root / "spider_eval/evaluation_bart.py").exists():
        eval_script = project_root / "spider_eval/evaluation_bart.py"

    cmd = [
        sys.executable,
        str(eval_script),
        "--gold", str(gold_sql_file),
        "--pred", str(pred_sql_file),
        "--etype", "exec",
        "--db", str(db_root),
        "--table", str(table_json),
    ]

    print(f"\nRunning Spider evaluation using {eval_script.name}...")
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")

    if proc.returncode != 0:
        print("\nSpider evaluation crashed.")
        print(proc.stderr)
        return

    print("\n--- Spider Eval Output ---")
    print("\n".join(proc.stdout.splitlines()[-20:]))

    acc = parse_exec_accuracy(proc.stdout)
    if acc is not None:
        print(f"\n🎯 Official Execution Accuracy: {acc*100:.2f}%")
    else:
        print("\nCould not parse official accuracy.")


if __name__ == "__main__":
    main()