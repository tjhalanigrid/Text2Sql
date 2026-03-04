# """
# TEXT2SQL ENGINE
# Loads RLHF/SFT model once and answers questions on any Spider database.
# """

# import sqlite3
# import torch
# import re
# from pathlib import Path
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from peft import PeftModel

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# DB_ROOT = PROJECT_ROOT / "data" / "database"


# class Text2SQLEngine:

#     # =========================
#     # INIT
#     # =========================
#     def __init__(
#         self,
#         adapter_path="checkpoints/best_rlhf_model",
#         base_model_name="Salesforce/codet5-base"
#     ):
#         self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

#         adapter_path = (PROJECT_ROOT / adapter_path).resolve()
#         if not adapter_path.exists():
#             raise FileNotFoundError(f"RLHF adapter not found at: {adapter_path}")

#         print(f"Using adapter: {adapter_path}\n")

#         # TOKENIZER
#         print("Loading tokenizer...")
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), local_files_only=True)
#         except Exception:
#             print("Adapter tokenizer missing — using base tokenizer")
#             self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

#         # MODEL
#         print("Loading base model...")
#         base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

#         print("Loading LoRA adapter...")
#         self.model = PeftModel.from_pretrained(base, str(adapter_path)).to(self.device)
#         self.model.eval()

#         print("✅ Model ready\n")

#     # =========================
#     # PROMPT
#     # =========================
#     def build_prompt(self, question, schema):
#         return f"""translate English to SQL:

# Schema:
# {schema}

# Question:
# {question}

# SQL:"""

#     # =========================
#     # SCHEMA
#     # =========================
#     def get_schema(self, db_id):
#         db_path = DB_ROOT / db_id / f"{db_id}.sqlite"

#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()

#         tables = cursor.execute(
#             "SELECT name FROM sqlite_master WHERE type='table';"
#         ).fetchall()

#         schema_lines = []

#         for (table,) in tables:
#             cols = cursor.execute(f"PRAGMA table_info({table});").fetchall()
#             col_names = [c[1] for c in cols]
#             schema_lines.append(f"{table}({', '.join(col_names)})")

#         conn.close()
#         return "\n".join(schema_lines)

#     # =========================
#     # GENERATE SQL
#     # =========================
#     def generate_sql(self, question, db_id):
#         schema = self.get_schema(db_id)
#         prompt = self.build_prompt(question, schema)

#         inputs = self.tokenizer(
#             prompt,
#             return_tensors="pt",
#             truncation=True,
#             max_length=512
#         ).to(self.device)

#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=128,
#                 num_beams=4,
#                 do_sample=False,
#                 repetition_penalty=1.1
#             )

#         sql = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

#         if "SQL:" in sql:
#             sql = sql.split("SQL:")[-1]

#         return sql.strip()

#     # =========================
#     # SQL REPAIR (semantic fixes)
#     # =========================
#     def repair_sql(self, question, sql):
#         q = question.lower()

#         # contains / with / include
#         if any(word in q for word in ["contain", "with", "include", "having"]):
#             sql = re.sub(r"=\s*'([^']+)'", r"LIKE '%\1%'", sql, flags=re.IGNORECASE)

#         # never purchased pattern (Spider classic)
#         if "never" in q or "no purchase" in q:
#             if "customer" in sql.lower() and "invoice" in sql.lower():
#                 sql = """
# SELECT c.FirstName
# FROM Customer c
# LEFT JOIN Invoice i ON c.CustomerId = i.CustomerId
# WHERE i.CustomerId IS NULL
# """

#         # title search case insensitive
#         if "title" in q:
#             sql = re.sub(r"=\s*'([^']+)'", r"LIKE '%\1%'", sql, flags=re.IGNORECASE)

#         return sql

#     # =========================
#     # CLEAN SQL
#     # =========================
#     def clean_sql(self, sql: str):
#         sql = sql.strip()

#         # SQLite prefers single quotes
#         sql = sql.replace('"', "'")

#         # Case-insensitive equality
#         # album.title = 'Balls to the Wall'
#         # → album.title LIKE 'Balls to the Wall' COLLATE NOCASE
#         sql = re.sub(
#             r"([\w\.]+)\s*=\s*'([^']+)'",
#             r"\1 LIKE '\2' COLLATE NOCASE",
#             sql,
#             flags=re.IGNORECASE
#         )

#         # convert exact LIKE → contains LIKE
#         # LIKE 'rock' → LIKE '%rock%'
#         sql = re.sub(
#             r"LIKE\s*'([^%][^']*[^%])'\s*COLLATE NOCASE",
#             r"LIKE '%\1%' COLLATE NOCASE",
#             sql,
#             flags=re.IGNORECASE
#         )

#         # prevent dumping whole DB
#         if "limit" not in sql.lower():
#             sql += " LIMIT 50"

#         return sql

#     # =========================
#     # EXECUTE
#     # =========================
#     def execute_sql(self, question, sql, db_id):
#         db_path = DB_ROOT / db_id / f"{db_id}.sqlite"

#         # semantic repair MUST use original question
#         sql = self.repair_sql(question, sql)
#         sql = self.clean_sql(sql)

#         print("\nFinal SQL:")
#         print(sql)

#         try:
#             conn = sqlite3.connect(db_path)
#             cursor = conn.cursor()

#             cursor.execute(sql)

#             rows = cursor.fetchall()
#             columns = [d[0] for d in cursor.description] if cursor.description else []

#             conn.close()
#             return columns, rows, None

#         except Exception as e:
#             return [], [], str(e)

#     # =========================
#     # FULL PIPELINE
#     # =========================
#     def ask(self, question, db_id):
#         sql = self.generate_sql(question, db_id)
#         cols, rows, error = self.execute_sql(question, sql, db_id)

#         return {
#             "question": question,
#             "sql": sql,
#             "columns": cols,
#             "rows": rows,
#             "error": error
#         }


# # =========================
# # SINGLETON INSTANCE
# # =========================
# _engine = None

# def get_engine():
#     global _engine
#     if _engine is None:
#         _engine = Text2SQLEngine()
#     return _engine



"""
TEXT2SQL ENGINE
Universal Spider-compatible execution engine
"""

import sqlite3
import torch
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = PROJECT_ROOT / "data" / "database"


class Text2SQLEngine:

    # =========================
    # INIT
    # =========================
    def __init__(self,
                 adapter_path="checkpoints/best_rlhf_model",
                 base_model_name="Salesforce/codet5-base"):

        self.device = "mps" if torch.backends.mps.is_available() else \
                      "cuda" if torch.cuda.is_available() else "cpu"

        adapter_path = (PROJECT_ROOT / adapter_path).resolve()
        if not adapter_path.exists():
            raise FileNotFoundError(f"RLHF adapter not found at: {adapter_path}")

        print(f"Using adapter: {adapter_path}\n")

        # tokenizer
        print("Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), local_files_only=True)
        except Exception:
            print("Adapter tokenizer missing — using base tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # model
        print("Loading base model...")
        base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

        print("Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(base, str(adapter_path)).to(self.device)
        self.model.eval()

        print("✅ Model ready\n")

    # =========================
    # PROMPT
    # =========================
    def build_prompt(self, question, schema):
        return f"""translate English to SQL:

Schema:
{schema}

Question:
{question}

SQL:"""

    # =========================
    # SCHEMA
    # =========================
    def get_schema(self, db_id):
        db_path = DB_ROOT / db_id / f"{db_id}.sqlite"

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()

        schema_lines = []

        for (table,) in tables:
            cols = cursor.execute(f"PRAGMA table_info({table});").fetchall()
            col_names = [c[1] for c in cols]
            schema_lines.append(f"{table}({', '.join(col_names)})")

        conn.close()
        return "\n".join(schema_lines)

    # =========================
    # GENERATE SQL
    # =========================
    def generate_sql(self, question, db_id):
        schema = self.get_schema(db_id)
        prompt = self.build_prompt(question, schema)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=4,
                do_sample=False
            )

        sql = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # remove prompt echo
        if "SQL:" in sql:
            sql = sql.split("SQL:")[-1]

        return sql.strip()

    # =========================
    # REPAIR SQL (generic)
    # =========================
    def repair_sql(self, question, sql):

        q = question.lower()
        s = sql.lower()

    # ----------------------------
    # NEGATION (never / no / without)
    # ----------------------------
        if any(word in q for word in ["never", "no ", "without"]):

        # find main table and joined table
            m = re.search(r"from\s+(\w+).*join\s+(\w+)", s)
            if m:
                left = m.group(1)
                right = m.group(2)

            # find join key
                key = re.search(r"on\s+(\w+\.\w+)\s*=\s*(\w+\.\w+)", s)
                if key:
                    left_key = key.group(1)
                    right_key = key.group(2)

                # build anti join automatically
                    sql = f"""
    SELECT {left}.*
    FROM {left}
    LEFT JOIN {right} ON {left_key} = {right_key}


    WHERE {right_key} IS NULL
"""

    # ----------------------------
    # CONTAINS queries
    # ----------------------------
        if any(w in q for w in ["contain", "with", "include"]):
            sql = re.sub(r"=\s*'([^']+)'", r"LIKE '%\1%'", sql, flags=re.IGNORECASE)

        return sql

    # =========================
    # CLEAN SQL (syntax safe)
    # =========================
    def clean_sql(self, sql: str):

        sql = sql.strip()

        # remove duplicate SQL:
        sql = re.sub(r"(SQL:)+", "", sql, flags=re.IGNORECASE)

        # keep only first statement
        sql = sql.split(";")[0]

        # double → single quotes
        sql = sql.replace('"', "'")

        # ONLY apply LIKE to text columns (skip numbers)
        sql = re.sub(
            r"=\s*'([A-Za-z][^']*)'",
            r"LIKE '\1' COLLATE NOCASE",
            sql,
            flags=re.IGNORECASE
        )

        # add LIMIT
        if "limit" not in sql.lower():
            sql += " LIMIT 50"

        return sql

    # =========================
    # EXECUTE
    # =========================
    def execute_sql(self, question, sql, db_id):

        db_path = DB_ROOT / db_id / f"{db_id}.sqlite"

        sql = self.repair_sql(question, sql)
        sql = self.clean_sql(sql)

        print("\nFinal SQL:")
        print(sql)

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute(sql)

            rows = cursor.fetchall()
            columns = [d[0] for d in cursor.description] if cursor.description else []

            conn.close()
            return sql , columns, rows, None

        except Exception as e:
            return sql ,[], [], str(e)

    # =========================
    # PIPELINE
    # =========================
    def ask(self, question, db_id):
        raw_sql = self.generate_sql(question, db_id)
        final_sql, cols, rows, error = self.execute_sql(question, raw_sql, db_id)

        return {
            "question": question,
            "sql": final_sql,   # now UI shows repaired SQL
            "columns": cols,
            "rows": rows,
            "error": error
        }


# singleton
_engine = None
def get_engine():
    global _engine
    if _engine is None:
        _engine = Text2SQLEngine()
    return _engine