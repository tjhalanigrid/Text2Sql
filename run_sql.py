import sqlite3
from datasets import load_dataset
import os

# load spider dataset (official loader downloads DB automatically)
dataset = load_dataset("spider")

example = dataset["train"][0]

question = example["question"]
query = example["query"]
db_id = example["db_id"]

print("\nQuestion:", question)
print("SQL:", query)
print("DB:", db_id)

# HF stores sqlite databases inside dataset cache directory
db_dir = example["db_path"]   # <-- THIS is the real path
print("\nDatabase path:", db_dir)

conn = sqlite3.connect(db_dir)
cursor = conn.cursor()

cursor.execute(query)
result = cursor.fetchall()

print("\nSQL RESULT:", result)

conn.close()

