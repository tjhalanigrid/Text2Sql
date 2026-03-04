

import sqlite3
import torch
import re
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from src.sql_validator import SQLValidator
from src.schema_encoder import SchemaEncoder  

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_ROOT = PROJECT_ROOT / "data" / "database"

# ==========================================
#  UNIVERSAL STRING NORMALIZERS
# ==========================================
def normalize_question(q: str):
    q = q.lower().strip()
    q = re.sub(r"distinct\s+(\d+)", r"\1 distinct", q)
    q = re.sub(r"\s+", " ", q)
    return q

def semantic_fix(question, sql):
    """Universal structural fixes that apply to ALL queries and ALL databases."""
    q = question.lower().strip()
    s = sql.lower()

    # UNIVERSAL LIMIT CATCHER: Enforce LIMIT if a number is in the question
    #  FIXED: Removed the '?'. Now it ONLY catches numbers explicitly preceded by "show", "top", "limit", etc.
    # This stops it from accidentally catching years like "2000".
    num_match = re.search(r'\b(?:show|list|top|limit|get|first|last)\s+(\d+)\b', q)
    if num_match and "limit" not in s and "count(" not in s:
        limit_val = num_match.group(1)
        sql = sql.rstrip(";")
        sql = f"{sql.strip()} LIMIT {limit_val}"

    return sql


class Text2SQLEngine:
    def __init__(self,
                 adapter_path="checkpoints/best_rlhf_model",
                 base_model_name="Salesforce/codet5-base",
                 use_lora=True):

        self.device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.validator = SQLValidator(DB_ROOT)
        self.schema_encoder = SchemaEncoder(DB_ROOT)
        self.schema_mode = "structured"
        
        # Security Keywords
        self.dml_keywords = r'\b(delete|update|insert|drop|alter|truncate)\b'

        print("Loading base model...")
        base = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

        if not use_lora:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.model = base.to(self.device)
            self.model.eval()
            print("✅ Base model ready\n")
            return

        adapter_path = (PROJECT_ROOT / adapter_path).resolve()
        
        print("Loading tokenizer and LoRA adapter...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        self.model = PeftModel.from_pretrained(base, str(adapter_path)).to(self.device)
        self.model.eval()
        print("✅ RLHF model ready\n")

    # ==========================================
    # ---------------- PROMPT BUILDERS ---------
    # ==========================================
    def build_prompt(self, question, schema):
        return f"""You are an expert SQL generator.
Database schema:
{schema}
Generate a valid SQLite query for the question.
Question:
{question}
SQL:
"""

    def build_repair_prompt(self, question, schema, bad_sql, error_msg):
        #  UNIVERSAL UPGRADE: Extract the hallucinated column and explicitly warn the model
        hallucinated_warning = ""
        col_match = re.search(r"no such column:\s*([^\s]+)", error_msg, re.IGNORECASE)
        if col_match:
            bad_col = col_match.group(1)
            hallucinated_warning = f"\n🚨 CRITICAL ERROR: You hallucinated the column '{bad_col}'. IT DOES NOT EXIST. Look at the schema and find the actual column name (it might be spelled differently or be a synonym like 'details', 'desc', or have a typo)."

        return f"""You are an expert SQL generator.
Database schema:
{schema}

You generated this incorrect SQL for the question "{question}":
{bad_sql}

Execution failed with this SQLite error:
{error_msg}{hallucinated_warning}

UNIVERSAL RULES TO FIX THIS:
1. NEVER invent or guess column names. Use ONLY the exact table and column names listed in the schema above.
2. Watch out for typos in the database schema! If you need 'assessment', look for 'asessment'. If you need 'name', look for 'details'.
3. If the error is "no such column", you either hallucinated the name, or you forgot an INNER JOIN. Check the schema and fix it.
4. If the query requires a COUNT() but also selects names, ensure you added a GROUP BY.

Write the corrected SQLite SQL query.
SQL:
"""

    def get_schema(self, db_id):
        return self.schema_encoder.structured_schema(db_id)

    # ==========================================
    # ---------------- SQL POSTPROCESS ---------
    # ==========================================
    def extract_sql(self, text: str):
        text = text.strip()
        if "SQL:" in text:
            text = text.split("SQL:")[-1]
        match = re.search(r"select[\s\S]*", text, re.IGNORECASE)
        if match:
            text = match.group(0)
        return text.split(";")[0].strip()

    def clean_sql(self, sql: str):
        sql = sql.replace('"', "'")
        sql = re.sub(r"\s+", " ", sql)
        return sql.strip()

    def repair_logic(self, question, sql):
        """Universal logical repairs (like missing NOT NULL for negation)"""
        q = question.lower()
        s = sql.lower()

        # Universal Negation Auto-Joiner
        if any(word in q for word in ["never", "no ", "without"]):
            m = re.search(r"from\s+(\w+).*join\s+(\w+)", s)
            if m:
                left, right = m.group(1), m.group(2)
                key = re.search(r"on\s+(\w+\.\w+)\s*=\s*(\w+\.\w+)", s)
                if key:
                    sql = f"SELECT {left}.* FROM {left} LEFT JOIN {right} ON {key.group(1)} = {key.group(2)} WHERE {key.group(2)} IS NULL"

        # Universal LIKE wildcard injection
        if any(w in q for w in ["contain", "with", "include"]):
            sql = re.sub(r"=\s*'([^']+)'", r"LIKE '%\1%'", sql, flags=re.IGNORECASE)

        return sql

    # ==========================================
    # ---------------- GENERATE ----------------
    # ==========================================
    def generate_sql(self, prompt, is_repair=False):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        #  FIXED: Dynamic Generation Parameters (No more terminal warnings)
        gen_kwargs = {
            "max_new_tokens": 128,
        }
        
        if is_repair:
            # If the model failed, it needs to think differently. 
            # We turn off rigid beam search and introduce sampling so it doesn't repeat the exact same broken SQL.
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = 0.5
            gen_kwargs["top_p"] = 0.9
        else:
            # First attempt is strictly deterministic for maximum benchmark accuracy
            gen_kwargs["num_beams"] = 5
            gen_kwargs["do_sample"] = False
            gen_kwargs["early_stopping"] = True # <--- Moved here so it doesn't clash with sampling!

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.clean_sql(self.extract_sql(decoded))

    # ==========================================
    # ---------------- EXECUTE -----------------
    # ==========================================
    def execute_sql(self, question, sql, db_id):
        
        # 🛡️ DEFENSE LAYER 2: Block Execution of Malicious SQL
        if re.search(self.dml_keywords, sql, re.IGNORECASE):
            return sql, [], [], "❌ Security Alert: Malicious DML/DDL SQL syntax blocked."

        db_path = DB_ROOT / db_id / f"{db_id}.sqlite"

        sql = self.repair_logic(question, sql)
        sql = self.clean_sql(sql)
        sql = semantic_fix(question, sql)

        is_valid, reason = self.validator.validate(sql, db_id)
        if not is_valid:
            return sql, [], [], f"Blocked unsafe SQL: {reason}"

        try:
            conn = sqlite3.connect(db_path)
            start_time = time.monotonic()
            def timeout_handler():
                return 1 if (time.monotonic() - start_time) > 5.0 else 0
            conn.set_progress_handler(timeout_handler, 10000)

            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            columns = [d[0] for d in cursor.description] if cursor.description else []
            conn.close()
            
            return sql, columns, rows, None

        except Exception as e:
            return sql, [], [], str(e)

    # ==========================================
    # ---------------- PIPELINE ----------------
    # ==========================================
    def ask(self, question, db_id):
        question = normalize_question(question)
        
        # 🛡️ DEFENSE LAYER 1: Block Malicious Natural Language Intent Early
        if re.search(self.dml_keywords, question, re.IGNORECASE):
            return {
                "question": question,
                "sql": "-- BLOCKED",
                "columns": [],
                "rows": [],
                "error": "❌ Security Alert: Malicious intent (DELETE/DROP/UPDATE) detected in the prompt."
            }

        # 1. First Pass Generation
        schema = self.get_schema(db_id)
        prompt = self.build_prompt(question, schema)
        
        # is_repair=False -> Uses strict Beam Search
        raw_sql = self.generate_sql(prompt, is_repair=False) 
        
        # 2. First Execution Attempt
        final_sql, cols, rows, error = self.execute_sql(question, raw_sql, db_id)
        
        # 🤖 3. UNIVERSAL AGENTIC SELF-CORRECTION LOOP
        if error and "Security Alert" not in error:
            print(f"\n Caught SQLite Error: {error}")
            print(f" Triggering Stochastic LLM Self-Correction...")
            
            # Feed the explicit error instructions back to the LLM
            repair_prompt = self.build_repair_prompt(question, schema, final_sql, error)
            
            # is_repair=True -> Uses Temperature Sampling to break out of hallucination loops
            repaired_sql = self.generate_sql(repair_prompt, is_repair=True) 
            
            # Try executing the repaired SQL
            final_sql, cols, rows, error = self.execute_sql(question, repaired_sql, db_id)
            
            if not error:
                print("✅ Universal Agent successfully self-corrected the query!")
            else:
                print("❌ Model failed self-correction.")
        
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