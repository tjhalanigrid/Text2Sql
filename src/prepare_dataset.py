import json
import os
import sqlite3
from datasets import Dataset
from transformers import T5Tokenizer

# =========================================================
# PROJECT ROOT (VERY IMPORTANT — fixes path issues)
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_JSON = os.path.join(BASE_DIR, "data", "train_spider.json")
DEV_JSON   = os.path.join(BASE_DIR, "data", "dev.json")
DB_FOLDER  = os.path.join(BASE_DIR, "data", "database")

SAVE_TRAIN = os.path.join(BASE_DIR, "data", "tokenized", "train")
SAVE_DEV   = os.path.join(BASE_DIR, "data", "tokenized", "validation")

os.makedirs(os.path.dirname(SAVE_TRAIN), exist_ok=True)

print("Project root:", BASE_DIR)
print("Train file:", TRAIN_JSON)
print("Database folder:", DB_FOLDER)

# =========================================================
# TOKENIZER
# =========================================================
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# =========================================================
# READ DATABASE SCHEMA
# =========================================================
def get_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    tables = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    ).fetchall()

    schema_text = []

    for table in tables:
        table = table[0]

        columns = cursor.execute(f"PRAGMA table_info({table});").fetchall()
        col_names = [c[1] for c in columns]

        schema_text.append(f"{table}({', '.join(col_names)})")

    conn.close()
    return "\n".join(schema_text)


# =========================================================
# BUILD TRAINING EXAMPLES
# =========================================================
def build_examples(spider_json):

    print(f"\nBuilding dataset from: {spider_json}")

    data = json.load(open(spider_json))

    inputs = []
    outputs = []

    for ex in data:

        question = ex["question"]
        sql = ex["query"]
        db_id = ex["db_id"]

        db_path = os.path.join(DB_FOLDER, db_id, f"{db_id}.sqlite")

        # skip if db missing (safety)
        if not os.path.exists(db_path):
            continue

        schema = get_schema(db_path)

        # ⭐ SCHEMA-AWARE PROMPT (VERY IMPORTANT)
        input_text = f"""Database Schema:
{schema}

Translate English to SQL:
{question}
SQL:
"""

        inputs.append(input_text)
        outputs.append(sql)

    return Dataset.from_dict({"input": inputs, "output": outputs})


# =========================================================
# TOKENIZE
# =========================================================
def tokenize(example):

    model_input = tokenizer(
        example["input"],
        max_length=512,
        padding="max_length",
        truncation=True
    )

    label = tokenizer(
        example["output"],
        max_length=256,
        padding="max_length",
        truncation=True
    )

    model_input["labels"] = label["input_ids"]
    return model_input


# =========================================================
# RUN PIPELINE
# =========================================================
print("\nBuilding TRAIN dataset...")
train_dataset = build_examples(TRAIN_JSON)

print("Tokenizing TRAIN dataset...")
tokenized_train = train_dataset.map(tokenize, batched=False)

print("Saving TRAIN dataset...")
tokenized_train.save_to_disk(SAVE_TRAIN)


print("\nBuilding VALIDATION dataset...")
val_dataset = build_examples(DEV_JSON)

print("Tokenizing VALIDATION dataset...")
tokenized_val = val_dataset.map(tokenize, batched=False)

print("Saving VALIDATION dataset...")
tokenized_val.save_to_disk(SAVE_DEV)

print("\nDONE ✔ Dataset prepared successfully!")
print("Train saved at:", SAVE_TRAIN)
print("Validation saved at:", SAVE_DEV)