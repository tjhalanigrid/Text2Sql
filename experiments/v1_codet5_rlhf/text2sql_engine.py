# import sqlite3
# import torch
# import re
# from pathlib import Path
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from peft import PeftModel
# from src.sql_validator import SQLValidator

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# DB_ROOT = PROJECT_ROOT / "data" / "database"


# class Text2SQLEngine:
#     def __init__(self, adapter_path="checkpoints/best_rlhf_model", base_model_name="Salesforce/codet5-base"):
#         self.device = "mps" if torch.backends.mps.is_available() else (
#             "cuda" if torch.cuda.is_available() else "cpu"
#         )

#         adapter_path = (PROJECT_ROOT / adapter_path).resolve()
#         if not adapter_path.exists():
#             raise FileNotFoundError(f"Adapter not found: {adapter_path}")

#         print("Loading tokenizer...")
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), local_files_only=True)
#         except Exception:
#             self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

#         print("Loading base model...")
#         base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

#         print("Loading LoRA adapter...")
#         self.model = PeftModel.from_pretrained(base, str(adapter_path)).to(self.device)
#         self.model.eval()
#         print("✅ Model ready\n")
        
#         self.validator = SQLValidator(DB_ROOT)



#     # def __init__(self,
#     #          adapter_path="checkpoints/rl_step_1800",
#     #          base_model_name="Salesforce/codet5-base",
#     #          use_lora=False):

#     #     self.device = "mps" if torch.backends.mps.is_available() else (
#     #         "cuda" if torch.cuda.is_available() else "cpu"
#     #     )

#     #     print("Loading tokenizer...")
#     #     self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

#     #     print("Loading base model...")
#     #     base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

#     # # ===== BASE MODEL MODE =====
#     #     if not use_lora:
#     #         print("⚪ Running BASE model (no RLHF)")
#     #         self.model = base.to(self.device)
#     #         self.model.eval()
#     #         print("✅ Base model ready\n")
#     #         return

#     # # ===== RLHF MODE =====
#     #     adapter_path = (PROJECT_ROOT / adapter_path).resolve()
#     #     if not adapter_path.exists():
#     #         raise FileNotFoundError(f"RLHF adapter not found at: {adapter_path}")

#     #     print("Loading LoRA adapter...")
#     #     self.model = PeftModel.from_pretrained(base, str(adapter_path)).to(self.device)
#     #     self.model.eval()

#     #     print("✅ RLHF model ready\n")
#     # ---------------- PROMPT ----------------
#     def build_prompt(self, question, schema):
#         return f"Translate English to SQL.\nSchema:\n{schema}\nQuestion: {question}\nSQL:"

#     # ---------------- SCHEMA ----------------
#     def get_schema(self, db_id):
#         db_path = DB_ROOT / db_id / f"{db_id}.sqlite"
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()

#         tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
#         schema_lines = []

#         for (table,) in tables:
#             cols = cursor.execute(f"PRAGMA table_info({table});").fetchall()
#             col_names = [c[1] for c in cols]
#             schema_lines.append(f"{table}({', '.join(col_names)})")

#         conn.close()
#         return "\n".join(schema_lines)

#     # ---------------- SQL POSTPROCESS ----------------
#     def extract_sql(self, text: str):
#         text = text.strip()

#         if "SQL:" in text:
#             text = text.split("SQL:")[-1]

#         match = re.search(r"select[\s\S]*", text, re.IGNORECASE)
#         if match:
#             text = match.group(0)

#         text = text.split(";")[0]
#         return text.strip()

#     def clean_sql(self, sql: str):
#         sql = sql.replace('"', "'")
#         sql = re.sub(r"\s+", " ", sql)
#         if not re.search(r"limit\s+\d+", sql, re.IGNORECASE):
#             sql += " LIMIT 50"
#         return sql

#     def repair_sql(self, question, sql):
#         """Repairs semantic mismatches based on the original question."""
#         q = question.lower()
#         s = sql.lower()

#         # ----------------------------
#         # NEGATION (never / no / without)
#         # ----------------------------
#         if any(word in q for word in ["never", "no ", "without"]):
#             m = re.search(r"from\s+(\w+).*join\s+(\w+)", s)
#             if m:
#                 left = m.group(1)
#                 right = m.group(2)
#                 key = re.search(r"on\s+(\w+\.\w+)\s*=\s*(\w+\.\w+)", s)
#                 if key:
#                     left_key = key.group(1)
#                     right_key = key.group(2)
#                     sql = f"""SELECT {left}.*
# FROM {left}
# LEFT JOIN {right} ON {left_key} = {right_key}
# WHERE {right_key} IS NULL"""

#         # ----------------------------
#         # CONTAINS queries
#         # ----------------------------
#         if any(w in q for w in ["contain", "with", "include"]):
#             sql = re.sub(r"=\s*'([^']+)'", r"LIKE '%\1%'", sql, flags=re.IGNORECASE)

#         return sql

#     # ---------------- GENERATE ----------------
#     def generate_sql(self, question, db_id):
#         schema = self.get_schema(db_id)
#         prompt = self.build_prompt(question, schema)

#         inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=128,
#                 num_beams=5,
#                 early_stopping=True,
#                 repetition_penalty=1.05,
#                 no_repeat_ngram_size=3
#             )

#         decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         sql = self.extract_sql(decoded)
#         return self.clean_sql(sql)

#     # ---------------- EXECUTE ----------------
#     def execute_sql(self, question, sql, db_id):
#         db_path = DB_ROOT / db_id / f"{db_id}.sqlite"

#         # 1️⃣ Repair logic (anti-join, contains, etc.)
#         sql = self.repair_sql(question, sql)

#         # 2️⃣ Clean syntax
#         sql = self.clean_sql(sql)

#         # 3️⃣ Validate before running (NEW SAFETY LAYER)
#         is_valid, reason = self.validator.validate(sql, db_id)

#         print("\nFinal SQL:")
#         print(sql)

#         if not is_valid:
#             return sql, [], [], f"Blocked unsafe SQL: {reason}"

#         # 4️⃣ Execute
#         try:
#             conn = sqlite3.connect(db_path)
#             cursor = conn.cursor()

#             cursor.execute(sql)

#             rows = cursor.fetchall()
#             columns = [d[0] for d in cursor.description] if cursor.description else []

#             conn.close()
#             return sql, columns, rows, None

#         except Exception as e:
#             return sql, [], [], str(e)

#     # ---------------- PIPELINE ----------------
#     def ask(self, question, db_id):
#         # 🛡️ 1. PRE-GENERATION SAFETY CHECK: Block malicious intent early
#         lower_q = question.lower()
#         dml_keywords = ["delete ", "update ", "insert ", "drop ", "alter ", "truncate "]
        
#         if any(keyword in lower_q for keyword in dml_keywords):
#             return {
#                 "question": question,
#                 "sql": "-- BLOCKED",
#                 "columns": [],
#                 "rows": [],
#                 "error": "Blocked unsafe SQL (DML/DDL detected in prompt)"
#             }

#         # 🤖 2. Generate SQL
#         raw_sql = self.generate_sql(question, db_id)
        
#         # ⚙️ 3. Execute and validate the generated SQL
#         final_sql, cols, rows, error = self.execute_sql(question, raw_sql, db_id)
        
#         return {
#             "question": question, 
#             "sql": final_sql, 
#             "columns": cols, 
#             "rows": rows, 
#             "error": error
#         }


# _engine = None

# def get_engine():
#     global _engine
#     if _engine is None:
#         _engine = Text2SQLEngine()
#     return _engine



import sqlite3
import torch
import re
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from src.sql_validator import SQLValidator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = PROJECT_ROOT / "data" / "database"


class Text2SQLEngine:
    def __init__(self,
                 adapter_path="checkpoints/rl_step_1800",
                 base_model_name="Salesforce/codet5-base",
                 use_lora=True):

        self.device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize safety validator
        self.validator = SQLValidator(DB_ROOT)

        print("Loading base model...")
        base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

        # ===== BASE MODEL MODE (For Evaluation) =====
        if not use_lora:
            print("⚪ Running BASE model (no RLHF)")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.model = base.to(self.device)
            self.model.eval()
            print("✅ Base model ready\n")
            return

        # ===== RLHF MODE (For UI / RLHF Evaluation) =====
        adapter_path = (PROJECT_ROOT / adapter_path).resolve()
        if not adapter_path.exists():
            raise FileNotFoundError(f"RLHF adapter not found at: {adapter_path}")

        print("Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        print("Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(base, str(adapter_path)).to(self.device)
        self.model.eval()

        print("✅ RLHF model ready\n")

    # ---------------- PROMPT ----------------
    def build_prompt(self, question, schema):
        return f"Translate English to SQL.\nSchema:\n{schema}\nQuestion: {question}\nSQL:"

    # ---------------- SCHEMA ----------------
    def get_schema(self, db_id):
        db_path = DB_ROOT / db_id / f"{db_id}.sqlite"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        schema_lines = []

        for (table,) in tables:
            cols = cursor.execute(f"PRAGMA table_info({table});").fetchall()
            col_names = [c[1] for c in cols]
            schema_lines.append(f"{table}({', '.join(col_names)})")

        conn.close()
        return "\n".join(schema_lines)

    # ---------------- SQL POSTPROCESS ----------------
    def extract_sql(self, text: str):
        text = text.strip()

        if "SQL:" in text:
            text = text.split("SQL:")[-1]

        match = re.search(r"select[\s\S]*", text, re.IGNORECASE)
        if match:
            text = match.group(0)

        text = text.split(";")[0]
        return text.strip()

    def clean_sql(self, sql: str):
        sql = sql.replace('"', "'")
        sql = re.sub(r"\s+", " ", sql)
        if not re.search(r"limit\s+\d+", sql, re.IGNORECASE):
            sql += " LIMIT 50"
        return sql

    def repair_sql(self, question, sql):
        """Repairs semantic mismatches based on the original question."""
        q = question.lower()
        s = sql.lower()

        # ----------------------------
        # NEGATION (never / no / without)
        # ----------------------------
        if any(word in q for word in ["never", "no ", "without"]):
            m = re.search(r"from\s+(\w+).*join\s+(\w+)", s)
            if m:
                left = m.group(1)
                right = m.group(2)
                key = re.search(r"on\s+(\w+\.\w+)\s*=\s*(\w+\.\w+)", s)
                if key:
                    left_key = key.group(1)
                    right_key = key.group(2)
                    sql = f"""SELECT {left}.*
FROM {left}
LEFT JOIN {right} ON {left_key} = {right_key}
WHERE {right_key} IS NULL"""

        # ----------------------------
        # CONTAINS queries
        # ----------------------------
        if any(w in q for w in ["contain", "with", "include"]):
            sql = re.sub(r"=\s*'([^']+)'", r"LIKE '%\1%'", sql, flags=re.IGNORECASE)

        return sql

    # ---------------- GENERATE ----------------
    def generate_sql(self, question, db_id):
        schema = self.get_schema(db_id)
        prompt = self.build_prompt(question, schema)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=5,
                early_stopping=True,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        sql = self.extract_sql(decoded)
        return self.clean_sql(sql)

    # ---------------- EXECUTE ----------------
    def execute_sql(self, question, sql, db_id):
        db_path = DB_ROOT / db_id / f"{db_id}.sqlite"

        # 1️⃣ Repair logic (anti-join, contains, etc.)
        sql = self.repair_sql(question, sql)

        # 2️⃣ Clean syntax
        sql = self.clean_sql(sql)

        # 3️⃣ Validate before running (NEW SAFETY LAYER)
        is_valid, reason = self.validator.validate(sql, db_id)

        print("\nFinal SQL:")
        print(sql)

        if not is_valid:
            return sql, [], [], f"Blocked unsafe SQL: {reason}"

        # 4️⃣ Execute with 5-second timeout
        try:
            conn = sqlite3.connect(db_path)
            
            # --- TIMEOUT LOGIC ---
            start_time = time.monotonic()
            def timeout_handler():
                return 1 if (time.monotonic() - start_time) > 5.0 else 0
            
            conn.set_progress_handler(timeout_handler, 10000)
            # ---------------------

            cursor = conn.cursor()
            cursor.execute(sql)

            rows = cursor.fetchall()
            columns = [d[0] for d in cursor.description] if cursor.description else []

            conn.close()
            return sql, columns, rows, None

        except sqlite3.OperationalError as e:
            if "interrupted" in str(e).lower():
                return sql, [], [], "⏳ Execution Error: Query timed out after 5 seconds."
            return sql, [], [], str(e)
        except Exception as e:
            return sql, [], [], str(e)

    # ---------------- PIPELINE ----------------
    def ask(self, question, db_id):
        # 🛡️ 1. PRE-GENERATION SAFETY CHECK: Block malicious intent early
        lower_q = question.lower()
        dml_keywords = ["delete ", "update ", "insert ", "drop ", "alter ", "truncate "]
        
        if any(keyword in lower_q for keyword in dml_keywords):
            return {
                "question": question,
                "sql": "-- BLOCKED",
                "columns": [],
                "rows": [],
                "error": "Blocked unsafe SQL (DML/DDL detected in prompt)"
            }

        # 🤖 2. Generate SQL
        raw_sql = self.generate_sql(question, db_id)
        
        # ⚙️ 3. Execute and validate the generated SQL
        final_sql, cols, rows, error = self.execute_sql(question, raw_sql, db_id)
        
        return {
            "question": question, 
            "sql": final_sql, 
            "columns": cols, 
            "rows": rows, 
            "error": error
        }


_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = Text2SQLEngine()
    return _engine