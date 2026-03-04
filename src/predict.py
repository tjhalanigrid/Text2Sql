import torch
import sqlite3
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------------------------------
# PATH
# --------------------------------------------------
MODEL_PATH = "outputs/model"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading fine-tuned model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.eval()

# --------------------------------------------------
# CONNECT DATABASE
# --------------------------------------------------
print("Connecting to database...")
# conn = sqlite3.connect("../data/database/department_management/department_management.sqlite")
conn = sqlite3.connect("data/database/department_management/department_management.sqlite")
cursor = conn.cursor()
print("Database connected ✔")

# --------------------------------------------------
# BUILD PROMPT
# --------------------------------------------------
def build_prompt(question):
    schema = """
Table department columns = Department_ID, Name, Creation, Ranking, Budget_in_Billions, Num_Employees.
Table head columns = head_ID, name, born_state, age.
Table management columns = department_ID, head_ID, temporary_acting.
"""
    return f"translate English to SQL: {schema} question: {question}"

# --------------------------------------------------
# GENERATE SQL
# --------------------------------------------------
def generate_sql(question):

    prompt = build_prompt(question)

    encoding = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            max_length=256,
            num_beams=5,
            early_stopping=True
        )

    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql.strip()

# --------------------------------------------------
# EVALUATE SQL (REWARD FUNCTION)
# --------------------------------------------------
def evaluate_sql(sql):
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()

        # executed but no useful result
        if len(rows) == 0:
            return -0.2, rows

        # good query
        else:
            return 1.0, rows

    except Exception as e:
        # invalid SQL
        return -1.0, str(e)

# --------------------------------------------------
# INTERACTIVE LOOP
# --------------------------------------------------
while True:
    q = input("\nAsk question (type exit to quit): ")

    if q.lower() == "exit":
        break

    sql = generate_sql(q)

    print("\nPredicted SQL:")
    print(sql)

    # ---------------- RUN SQL + REWARD ----------------
    reward, output = evaluate_sql(sql)

    print("\nReward:", reward)

    if reward == -1.0:
        print("SQL Error:", output)

    elif reward == -0.2:
        print("No results found")

    else:
        print("\nAnswer:")
        for r in output:
            print(r)
