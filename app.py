
"""
GRADIO DEMO UI
NL → SQL → Result Table
"""

import gradio as gr
import pandas as pd
import re
from src.text2sql_engine import get_engine

engine = get_engine()

# =========================
# SAMPLE QUESTIONS DATA
# =========================
# Tuple format: ("Question", "Database_ID")
SAMPLES = [
    ("Show 10 distinct employee first names.", "chinook_1"),
    ("Which artist has the most albums?", "chinook_1"),
    ("List all the tracks that belong to the 'Rock' genre.", "chinook_1"),
    ("What are the names of all the cities?", "flight_1"),
    ("Find the flight number and cost of the cheapest flight.", "flight_1"),
    ("List the airlines that fly out of New York.", "flight_1"),
    ("Which campus was opened between 1935 and 1939?", "csu_1"),
    
    # ("Find the building, room number, semester and year of all courses offered by Psychology.", "college_2"),
    ("Count the number of students in each department.", "college_2"),
    # ("List all student names and their assessment scores.", "student_assessment"),
    # ("Which student has the highest total score?", "student_assessment"),
    # ("Show the titles of all books published after 2000.", "book_2"),
    # ("Who is the author of the book with the highest price?", "book_2"),
    ("List the names of all clubs.", "club_1"),
    ("How many members does each club have?", "club_1"),
    ("Show the names of all cinemas.", "cinema"),
    ("Which cinema has the most screens?", "cinema")
    # ("List the names of all car makers.", "car_1"),
    # ("What is the average horsepower of cars produced in 2000?", "car_1"),
    # ("Show the names of all authors and the number of publications they have.", "academic"),
    # ("Find the names and prices of all products in the electronics department.", "department_store")
]

# Extract just the questions for the dropdown
SAMPLE_QUESTIONS = [q[0] for q in SAMPLES]

# =========================
# CORE FUNCTIONS
# =========================
def run_query(question, db_id):
    if not question.strip():
        return "", None, " Please enter a question."

    result = engine.ask(question, db_id)
    final_sql = result["sql"]

    # Error handling
    if result["error"]:
        return final_sql, None, f"❌ SQL Error:\n{result['error']}"

    #  UPGRADE 1: Elegant handling for ZERO ROWS (Null Result)
    if not result["rows"]:
        # Return an empty dataframe with headers if possible, instead of a broken table
        df = pd.DataFrame(columns=result.get("columns", []))
        explanation = "✅ Query executed successfully\n\nRows returned: 0\n\n Note: The query ran perfectly, but there are no matching records (null/empty result) in the database for this question."
        return final_sql, df, explanation

    # Convert to Pandas DataFrame for a beautiful UI table
    df = pd.DataFrame(result["rows"], columns=result["columns"])
    actual_rows = len(result["rows"])

    explanation = f"✅ Query executed successfully\n\nRows returned: {actual_rows}\n"

    #  UPGRADE 2: Check if they asked for a LIMIT, but got fewer rows
    limit_match = re.search(r'LIMIT\s+(\d+)', final_sql, re.IGNORECASE)
    if limit_match:
        requested_limit = int(limit_match.group(1))
        if actual_rows < requested_limit:
            explanation += f"\nℹ️ Note: The query allowed up to {requested_limit} results, but only found {actual_rows} matching records in the database.\n"

    explanation += """
This shows the model understood:
• Database schema
• Table relationships
• Query intent
"""
    return final_sql, df, explanation

def load_sample(selected_question):
    """Automatically updates the textbox and database dropdown when a sample is picked."""
    if not selected_question:
        return gr.update(), gr.update()
    
    # Find the matching database for the selected question
    db = next((db for q, db in SAMPLES if q == selected_question), "chinook_1")
    return gr.update(value=selected_question), gr.update(value=db)

def clear_inputs():
    """Resets the UI fields."""
    return gr.update(value=None), gr.update(value=""), gr.update(value="chinook_1"), "", None, ""

# =========================
# UI LAYOUT
# =========================
with gr.Blocks(theme=gr.themes.Soft(), title="Text-to-SQL RLHF") as demo:

    gr.Markdown(
        """
        #  Text-to-SQL using RLHF + Execution Reward
        Convert Natural Language to SQL, strictly validated and safely executed on local SQLite databases.
        """
    )

    # ===== TRAINED DATABASES =====
    DBS = sorted([
        "flight_1", "student_assessment", "store_1", "bike_1", "book_2", "chinook_1",
        "academic", "aircraft", "car_1", "cinema", "club_1", "csu_1",
        "college_1", "college_2", "company_1", "company_employee",
        "customer_complaints", "department_store", "employee_hire_evaluation",
        "museum_visit", "products_for_hire", "restaurant_1",
        "school_finance", "shop_membership", "small_bank_1",
        "soccer_1", "student_1", "tvshow", "voter_1", "world_1"
    ])

    with gr.Row():
        
        # --- LEFT COLUMN (Inputs & Samples) ---
        with gr.Column(scale=1):
            gr.Markdown("### 1. Configuration & Input")
            
            sample_dropdown = gr.Dropdown(
                choices=SAMPLE_QUESTIONS,
                label=" Quick Select a Sample Question",
                info="Picking a question will automatically select the right database!"
            )

            gr.Markdown("---")
            
            db_id = gr.Dropdown(
                choices=DBS,
                value="chinook_1",
                label="Select Database",
                interactive=True
            )

            question = gr.Textbox(
                label="Ask a Question",
                placeholder="Type your own question or select a sample above...",
                lines=3
            )

            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                run_btn = gr.Button(" Generate & Run SQL", variant="primary")

        # --- RIGHT COLUMN (Outputs) ---
        with gr.Column(scale=2):
            gr.Markdown("### 2. Execution Results")
            
            final_sql = gr.Code(language="sql", label="Final Executed SQL")
            
            result_table = gr.Dataframe(
                label="Query Result Table",
                interactive=False,
                wrap=True
            )

            explanation = gr.Textbox(label="Execution Details", lines=6)

    # =========================
    # EVENT LISTENERS
    # =========================
    # When a sample question is selected from the dropdown, update the textbox and DB
    sample_dropdown.change(
        fn=load_sample,
        inputs=[sample_dropdown],
        outputs=[question, db_id]
    )

    # Run the query
    run_btn.click(
        fn=run_query,
        inputs=[question, db_id],
        outputs=[final_sql, result_table, explanation]
    )

    # Clear button action
    clear_btn.click(
        fn=clear_inputs,
        inputs=[],
        outputs=[sample_dropdown, question, db_id, final_sql, result_table, explanation]
    )

if __name__ == "__main__":
    demo.launch()