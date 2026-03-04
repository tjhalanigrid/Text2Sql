from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from trl import AutoModelForSeq2SeqLMWithValueHead

import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from src.execution_reward import execution_reward  # noqa: E402


BASE_MODEL = os.environ.get("BASE_MODEL", "t5-small")
DB_ROOT = os.path.join(PROJECT_ROOT, "data", "database")

# Prefer RL best model if present; otherwise fall back.
RL_DIR = os.path.join(PROJECT_ROOT, "outputs", "rlhf_text2sql", "best_model")
if not os.path.isdir(RL_DIR):
    RL_DIR = os.path.join(PROJECT_ROOT, "outputs", "rlhf_text2sql")

SPLIT = "train[:100]"  # quick sanity check
MAX_NEW_TOKENS = 128

PREFIX = "translate English to SQL:"
MAX_SCHEMA_CHARS = 1500
MAX_INPUT_TOKENS = 512


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)


def get_db_path(db_id: str) -> str:
    return os.path.join(DB_ROOT, db_id, f"{db_id}.sqlite")


_SCHEMA_CACHE: Dict[str, str] = {}


def get_db_schema_text(db_path: str) -> str:
    if db_path in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[db_path]
    schema_text = ""
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            cur = conn.cursor()
            tables = cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            ).fetchall()
            for (tname,) in tables:
                cols = cur.execute(f'PRAGMA table_info(\"{tname}\")').fetchall()
                col_names = [c[1] for c in cols if c and isinstance(c[1], str)]
                schema_text += f"{tname}({', '.join(col_names)}) "
    except Exception:
        schema_text = ""
    if len(schema_text) > MAX_SCHEMA_CHARS:
        schema_text = schema_text[:MAX_SCHEMA_CHARS]
    _SCHEMA_CACHE[db_path] = schema_text
    return schema_text


def encode_prompt(tokenizer, question: str, schema: str) -> torch.Tensor:
    schema = (schema or "")[:MAX_SCHEMA_CHARS]
    prefix_schema = f"{PREFIX}\n\nSchema:\n"
    mid = "\n\nQuestion:\n"
    suffix = f"{question}\n\nSQL:"

    prefix_ids = tokenizer.encode(prefix_schema, add_special_tokens=False)
    schema_ids = tokenizer.encode(schema, add_special_tokens=False)
    mid_ids = tokenizer.encode(mid, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)

    eos_id = tokenizer.eos_token_id
    max_without_eos = MAX_INPUT_TOKENS - (1 if eos_id is not None else 0)

    fixed_len = len(prefix_ids) + len(mid_ids) + len(suffix_ids)
    if fixed_len > max_without_eos:
        keep = max(0, max_without_eos - (len(prefix_ids) + len(mid_ids)))
        suffix_ids = suffix_ids[:keep]
        fixed_len = len(prefix_ids) + len(mid_ids) + len(suffix_ids)

    remaining_for_schema = max_without_eos - fixed_len
    if remaining_for_schema < 0:
        remaining_for_schema = 0
    schema_ids = schema_ids[:remaining_for_schema]

    ids = (prefix_ids + schema_ids + mid_ids + suffix_ids)[:max_without_eos]
    if eos_id is not None:
        ids = ids + [eos_id]

    return torch.tensor(ids, dtype=torch.long).to(device)


def load_model_and_tokenizer():
    # Try loading the PPO-saved value-head model directly.
    try:
        tok = AutoTokenizer.from_pretrained(RL_DIR)
        mdl = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(RL_DIR).to(device)
        return tok, mdl
    except Exception:
        pass

    # Fallback: treat RL_DIR as a LoRA adapter directory.
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)
    try:
        base = PeftModel.from_pretrained(base, RL_DIR)
    except Exception:
        # Final fallback: use SFT adapter (if RL adapter not found)
        sft_dir = os.path.join(PROJECT_ROOT, "checkpoints", "sft_adapter")
        base = PeftModel.from_pretrained(base, sft_dir)
    return tok, base


def main() -> None:
    tokenizer, model = load_model_and_tokenizer()
    model.eval()

    ds = load_dataset("spider", split=SPLIT)

    correct = 0
    valid = 0

    for i, ex in enumerate(ds, start=1):
        question = ex["question"]
        gold_sql = ex["query"]
        db_id = ex["db_id"]
        db_path = get_db_path(db_id)
        schema = get_db_schema_text(db_path)

        inp = encode_prompt(tokenizer, question, schema)
        with torch.no_grad():
            out = model.generate(
                input_ids=inp.unsqueeze(0),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        pred_sql = tokenizer.decode(out[0], skip_special_tokens=True)
        r = execution_reward(pred_sql, db_path, gold_sql)
        if r > -1.0:
            valid += 1
        if r >= 1.0:
            correct += 1

        if i % 25 == 0:
            print(f"Evaluated {i}/{len(ds)}")

    n = len(ds)
    print("\nRESULTS")
    print(f"examples: {n}")
    print(f"execution_accuracy: {correct/n:.3f}")
    print(f"valid_sql_rate: {valid/n:.3f}")


if __name__ == "__main__":
    main()
