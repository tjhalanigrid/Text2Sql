import argparse
import random
import sqlite3
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    schema = ""
    for (table,) in tables:
        cols = cursor.execute(f"PRAGMA table_info({table});").fetchall()
        col_names = [c[1] for c in cols]
        schema += f"{table}({', '.join(col_names)})\n"

    conn.close()
    return schema

def build_prompt(question, schema):
    return f"""translate English to SQL:

Question: {question}

Schema:
{schema}

SQL:"""

def generate_sql(model, tokenizer, question, db_path):
    schema = load_schema(db_path)
    prompt = build_prompt(question, schema)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql.split("SQL:")[-1].strip()

def main(args):
    BASE_MODEL = "t5-small"   # same model you used in SFT

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)

    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()
    dataset = load_dataset("spider", split="validation")
    samples = random.sample(list(dataset), args.num_questions)

    for ex in samples:
        db_id = ex["db_id"]
        question = ex["question"]
        db_path = f"{args.db_path}/{db_id}/{db_id}.sqlite"

        print("\n==============================")
        print("DB:", db_id)
        print("Q :", question)

        try:
            sql = generate_sql(model, tokenizer, question, db_path)
            print("SQL:", sql)
        except Exception as e:
            print("ERROR:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--db_path", required=True)
    parser.add_argument("--num_questions", type=int, default=10)
    args = parser.parse_args()
    main(args)
