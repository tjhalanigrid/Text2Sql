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


# ---------------- PARSE ACC ----------------
def _parse_exec_accuracy(stdout: str) -> float | None:
    for line in stdout.splitlines():
        if line.strip().startswith("execution"):
            try:
                return float(line.split()[-1])
            except:
                return None
    return None


# ---------------- CLEAN SQL ----------------
def clean_prediction(pred_sql: str) -> str:
    pred_sql = pred_sql.strip()

    if "SQL:" in pred_sql:
        pred_sql = pred_sql.split("SQL:")[-1]

    pred_sql = pred_sql.replace('"', "'")
    pred_sql = re.sub(r"\s+", " ", pred_sql).strip()

    if not pred_sql.endswith(";"):
        pred_sql += ";"

    return pred_sql


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default="checkpoints/sft_t5")
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    adapter_dir = project_root / args.adapter

    db_root = project_root / "data/database"
    table_json = project_root / "data/tables.json"
    dev_json = project_root / "data/dev.json"
    gold_sql = project_root / "data/dev_gold.sql"
    pred_path = project_root / "pred.sql"

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter dir: {adapter_dir}")

    # ---------------- DEVICE ----------------
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    # ---------------- LOAD MODEL ----------------
    BASE_MODEL = "t5-small"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)

    model = PeftModel.from_pretrained(base, str(adapter_dir)).to(device)
    model = model.merge_and_unload()
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------- LOAD DATA ----------------
    with dev_json.open() as f:
        dev = json.load(f)[: args.num_samples]

    print("Generating predictions...\n")

    correct = 0
    total = len(dev)

    # ---------------- GENERATE + LIVE EXEC ----------------
    with pred_path.open("w") as out_f, torch.no_grad():

        for i, ex in enumerate(dev, start=1):

            db_id = ex["db_id"]
            question = ex["question"]
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
                early_stopping=True,
            )

            pred_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred_sql = clean_prediction(pred_sql)

            out_f.write(pred_sql + "\n")

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

                if sorted(pred_rows) == sorted(gold_rows):
                    correct += 1

            except Exception:
                pass  # execution failed

            # 🔥 PRINT EVERY 10
            if i % 10 == 0 or i == total:
                current_acc = correct / i
                print(f"{i}/{total} | Acc: {current_acc:.3f}")

    print("\nGeneration finished.\n")

    # ---------------- SPIDER EVAL ----------------
    eval_script = project_root / "spider_eval/evaluation.py"

    cmd = [
        sys.executable,
        str(eval_script),
        "--gold", str(gold_sql),
        "--pred", str(pred_path),
        "--etype", "exec",
        "--db", str(db_root),
        "--table", str(table_json),
    ]

    print("Running Spider evaluation...")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    print(proc.stdout)

    exec_acc = _parse_exec_accuracy(proc.stdout)
    if exec_acc is not None:
        print(f"\n🎯 Official Execution Accuracy: {exec_acc*100:.2f}%")
    else:
        print("Could not parse accuracy.")


if __name__ == "__main__":
    main()