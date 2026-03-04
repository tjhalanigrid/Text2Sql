# import gradio as gr
# import pandas as pd
# from src.text2sql_engine import get_engine

# engine = get_engine()

# # ----------------------------
# # Run query
# # ----------------------------
# def ask_db(question, db_id):
#     if not question.strip():
#         return "", "Ask something 😄", None

#     result = engine.ask(question, db_id)

#     # error
#     if result["error"]:
#         return result["sql"], f"❌ SQL Error:\n{result['error']}", None

#     # empty
#     if not result["rows"]:
#         return result["sql"], "⚠️ Query ran but returned no rows", None

#     # table
#     df = pd.DataFrame(result["rows"], columns=result["columns"])
#     return result["sql"], "✅ Success", df


# # ----------------------------
# # Available DBs
# # ----------------------------
# DBS = [
#     "flight_1","student_assessment","store_1","bike_1","book_2","chinook_1",
#     "academic","aircraft","car_1","cinema","club_1","csu_1"
# ]

# # ----------------------------
# # UI
# # ----------------------------
# with gr.Blocks(title="Text2SQL RLHF Demo") as demo:

#     gr.Markdown("# 🧠 Text-to-SQL using RLHF + Execution Reward")
#     gr.Markdown("Ask questions in English → Model writes SQL → Executes on database")

#     with gr.Row():
#         db = gr.Dropdown(DBS, value="chinook_1", label="Database")

#     question = gr.Textbox(
#         label="Ask a question",
#         placeholder="Example: Which artist has the most albums?"
#     )

#     run = gr.Button("Run Query")

#     sql_output = gr.Code(label="Generated SQL", language="sql")
#     status = gr.Textbox(label="Status")
#     table = gr.Dataframe(label="Query Result")

#     run.click(ask_db, inputs=[question, db], outputs=[sql_output, status, table])

# demo.launch()

"""
GRADIO DEMO UI
NL → SQL → Fixed SQL → Result
"""

import gradio as gr
from src.text2sql_engine import get_engine

engine = get_engine()

# =========================
# CORE FUNCTION
# =========================
def run_query(question, db_id):

    if not question.strip():
        return "", "", "", "⚠️ Please enter a question."

    result = engine.ask(question, db_id)

    generated_sql = result["sql"]
    fixed_sql = result.get("final_sql", generated_sql)

    # Error handling
    if result["error"]:
        return generated_sql, fixed_sql, "", f"❌ SQL Error:\n{result['error']}"

    # Empty result
    if not result["rows"]:
        return generated_sql, fixed_sql, "", "⚠️ Query ran but returned no rows"

    # Format table
    header = " | ".join(result["columns"])
    rows = "\n".join(" | ".join(str(x) for x in r) for r in result["rows"][:20])
    table = header + "\n" + "-" * len(header) + "\n" + rows

    explanation = f"""
✅ Query executed successfully

Rows returned: {len(result["rows"])}

This shows the model understood:
• Database schema
• Table relationships
• Query intent
"""

    return generated_sql, fixed_sql, table, explanation


# =========================
# UI
# =========================
with gr.Blocks(title="Text-to-SQL RLHF") as demo:

    gr.Markdown("# 🧠 Text-to-SQL using RLHF + Execution Reward")
    gr.Markdown(
        "Model trained across multiple relational databases.\n"
        "It learns SQL by executing queries and receiving correctness reward."
    )

    # ===== TRAINED DATABASES =====
    DBS = [
         "flight_1","student_assessment","store_1","bike_1","book_2","chinook_1",
        "academic","aircraft","car_1","cinema","club_1","csu_1",

    # medium difficulty (NEW)
        "college_1","college_2","company_1","company_employee",
        "customer_complaints","department_store","employee_hire_evaluation",
        "museum_visit","products_for_hire","restaurant_1",
        "school_finance","shop_membership","small_bank_1",
        "soccer_1","student_1","tvshow","voter_1","world_1"
    ]

    with gr.Row():
        db_id = gr.Dropdown(
            choices=DBS,
            value="chinook_1",
            label="Database (trained in RLHF)"
        )

    question = gr.Textbox(
        label="Ask a question",
        placeholder="Example: Which artist has the most albums?"
    )

    run_btn = gr.Button("Run Query", variant="primary")

    gr.Markdown("## 🔹 Model Generated SQL")
    gen_sql = gr.Code(language="sql")

    gr.Markdown("## 🔹 Corrected SQL (Execution Safe)")
    fixed_sql = gr.Code(language="sql")

    gr.Markdown("## 🔹 Query Result")
    result_table = gr.Textbox(lines=15)

    gr.Markdown("## 🔹 Explanation")
    explanation = gr.Textbox(lines=6)

    run_btn.click(
        fn=run_query,
        inputs=[question, db_id],
        outputs=[gen_sql, fixed_sql, result_table, explanation]
    )

if __name__ == "__main__":
    demo.launch()