import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE_MODEL = "Salesforce/codet5-base"
ADAPTER = "checkpoints/sft_adapter"   # change if needed

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(model, ADAPTER)

model = model.to(device)
model.eval()

# 5 random Spider style questions
questions = [
    "List all employee names",
    "Find the number of students in each department",
    "Show the average salary of employees",
    "Which flights depart from LA?",
    "Find customers who bought more than 5 items"
]

for q in questions:
    prompt = f"Translate to SQL: {q}"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.0,   # deterministic
        )

    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nQUESTION:", q)
    print("SQL:", sql)
    print("-"*60)
