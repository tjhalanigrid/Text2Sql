import json
import sqlite3
import torch
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = PROJECT_ROOT / "data" / "database"

# -------------------------------
# Extract SQL components
# -------------------------------
def extract_components(sql):
    sql = sql.lower()
    return {
        "select": "select" in sql,
        "where": "where" in sql,
        "group": "group by" in sql,
        "order": "order by" in sql,
        "and_or": (" and " in sql) or (" or " in sql),
        "join": "join" in sql
    }

# -------------------------------
# Fallback Difficulty Estimator
# -------------------------------
def estimate_difficulty(sql):
    """Fallback if 'difficulty' is missing from the JSON."""
    sql = sql.lower()
    joins = sql.count("join")
    conditions = sql.count("and") + sql.count("or")
    
    if "intersect" in sql or "except" in sql or "union" in sql or joins > 2:
        return "extra"
    elif joins == 2 or ("group by" in sql and conditions > 0):
        return "hard"
    elif joins == 1 or "group by" in sql or "order by" in sql:
        return "medium"
    else:
        return "easy"

# -------------------------------
# Load schema
# -------------------------------
def load_schema(db_path):
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode(errors='ignore')
    cursor = conn.cursor()

    tables = cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    ).fetchall()

    schema = ""
    for (table,) in tables:
        cols = cursor.execute(f"PRAGMA table_info({table});").fetchall()
        col_names = [c[1] for c in cols]
        schema += f"{table}({', '.join(col_names)})\n"

    conn.close()
    return schema

# -------------------------------
# Prompt
# -------------------------------
def build_prompt(question, schema):
    return f"""Database Schema:
{schema}

Translate English to SQL:
{question}
SQL:
"""

# -------------------------------
# Main
# -------------------------------
def main():
    adapter = "checkpoints/rl_step_1800"
    base_model = "Salesforce/codet5-base"

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizer and models...")
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model).to(device)
    model = PeftModel.from_pretrained(base, adapter).to(device)
    model = model.merge_and_unload()
    model.eval()

    dev_json = PROJECT_ROOT / "data" / "dev.json"

    with open(dev_json) as f:
        dev = json.load(f)[:1000]  # Adjust number to test more/less

    components_list = ["select", "where", "group", "order", "and_or", "join"]
    difficulties_list = ["easy", "medium", "hard", "extra"]

    # Nested dictionary for components
    stats = {
        comp: {diff: {"correct": 0, "total": 0} for diff in difficulties_list}
        for comp in components_list
    }

    # 🚀 NEW: Trackers for OVERALL accuracy by difficulty
    overall_correct = {diff: 0 for diff in difficulties_list}
    overall_total = {diff: 0 for diff in difficulties_list}

    print(f"\nRunning grouped evaluation on {len(dev)} examples...\n")

    for i, ex in enumerate(dev, 1):
        question = ex["question"]
        gold_sql = ex["query"]
        db_id = ex["db_id"]
        
        # Determine difficulty
        difficulty = ex.get("difficulty", estimate_difficulty(gold_sql))
        if difficulty not in difficulties_list:
            difficulty = "medium"

        db_path = DB_ROOT / db_id / f"{db_id}.sqlite"
        schema = load_schema(db_path)
        prompt = build_prompt(question, schema)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,
                num_beams=4,
                do_sample=False
            )

        pred_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "SQL:" in pred_sql:
            pred_sql = pred_sql.split("SQL:")[-1]

        # --- 1. Update Overall Accuracy Trackers ---
        overall_total[difficulty] += 1
        # Simple string match for quick overall accuracy
        if pred_sql.strip().lower() == gold_sql.strip().lower():
            overall_correct[difficulty] += 1

        # --- 2. Update Component Stats ---
        pred_comp = extract_components(pred_sql)
        gold_comp = extract_components(gold_sql)

        for comp in components_list:
            if gold_comp[comp]:  # If the gold SQL required this component
                stats[comp][difficulty]["total"] += 1
                if pred_comp[comp]: # If the model successfully generated it
                    stats[comp][difficulty]["correct"] += 1

        if i % 20 == 0:
            print(f"Processed {i}/{len(dev)}")

    # -------------------------------
    # Plotting (Grouped Bar Chart)
    # -------------------------------
    x = np.arange(len(components_list))
    width = 0.2

    def get_acc(diff):
        return [
            (stats[comp][diff]["correct"] / stats[comp][diff]["total"] * 100) if stats[comp][diff]["total"] > 0 else 0 
            for comp in components_list
        ]

    acc_easy = get_acc("easy")
    acc_medium = get_acc("medium")
    acc_hard = get_acc("hard")
    acc_extra = get_acc("extra")

    fig, ax = plt.subplots(figsize=(14, 7))

    bars1 = ax.bar(x - 1.5 * width, acc_easy, width, label='Easy', color='#2ecc71')
    bars2 = ax.bar(x - 0.5 * width, acc_medium, width, label='Medium', color='#f1c40f')
    bars3 = ax.bar(x + 0.5 * width, acc_hard, width, label='Hard', color='#e67e22')
    bars4 = ax.bar(x + 1.5 * width, acc_extra, width, label='Extra', color='#e74c3c')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('SQL Component Match Accuracy by Difficulty Level', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in components_list], fontsize=11)
    ax.legend(title="Query Difficulty")
    ax.set_ylim(0, 115)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, rotation=90)

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    autolabel(bars4)

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("component_by_difficulty_plot.png", dpi=300)

    # -------------------------------
    # 🚀 Terminal Printout
    # -------------------------------
    print("\n✅ Saved merged plot -> component_by_difficulty_plot.png")
    
    print("\n========================================")
    print("🏆 OVERALL AVERAGE ACCURACY BY DIFFICULTY")
    print("========================================")
    for diff in difficulties_list:
        if overall_total[diff] > 0:
            avg = round((overall_correct[diff] / overall_total[diff]) * 100, 2)
            print(f"{diff.capitalize():<8}: {avg:>5}%  ({overall_correct[diff]}/{overall_total[diff]} queries)")
        else:
            print(f"{diff.capitalize():<8}:   N/A  (0 queries)")
    print("========================================\n")

if __name__ == "__main__":
    main()