import json
from datasets import load_dataset

print("Loading Spider dataset...")
dataset = load_dataset("spider", split="train")

data = []

for ex in dataset:
    data.append({
        "question": ex["question"],
        "query": ex["query"],
        "db_id": ex["db_id"]   # ⭐ CRITICAL FIELD
    })

print("Saving JSON...")
with open("data/train_spider.json", "w") as f:
    json.dump(data, f, indent=2)

print("Done! File saved at data/train_spider.json")
