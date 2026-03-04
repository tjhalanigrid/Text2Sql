# =========================================================
# RLHF TRAINING FOR TEXT2SQL (STABLE PPO VERSION)
# =========================================================
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from peft import PeftModel
import os, sys, sqlite3, re, random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from execution_reward import execution_reward, extract_tables, extract_columns
try:
    import sqlparse  # gate PPO updates on parsable SQL only
except Exception:  # pragma: no cover
    sqlparse = None

# ======================================================
# DEVICE
# ======================================================
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ======================================================
# TRAINING SETTINGS
# ======================================================
NUM_EPOCHS = 15            
LOG_EVERY = 20
USE_SCHEMA = True
SCHEMA_WARMUP_EPOCHS = 2   
MAX_SCHEMA_CHARS = 1500
MAX_OUTPUT_TOKENS = 120    
ROLLOUTS_PER_EPOCH = 1024  

# ======================================================
# PATHS
# ======================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs/rlhf_text2sql")
output_dir = RL_MODEL_PATH
DB_ROOT = os.path.join(PROJECT_ROOT, "data/database")

ADAPTER_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "checkpoints/sft_adapter_codet5"))
FALLBACK_ADAPTER_PATH = ADAPTER_PATH
FALLBACK_ADAPTER_PATH_2 = os.path.join(PROJECT_ROOT, "checkpoints")

BASE_MODEL = os.environ.get("BASE_MODEL", "Salesforce/codet5-base")

# ======================================================
# LOAD MODEL (LoRA)
# ======================================================
def find_valid_adapter(path_candidates):
    for p in path_candidates:
        if p and os.path.exists(os.path.join(p, "adapter_config.json")):
            return os.path.abspath(p)
    return None

print("Loading base:", BASE_MODEL)

ADAPTER_PATH = find_valid_adapter([
    ADAPTER_PATH,
    FALLBACK_ADAPTER_PATH,
    FALLBACK_ADAPTER_PATH_2,
])

if ADAPTER_PATH is None:
    raise RuntimeError("❌ No valid LoRA adapter found!")

print("Loading adapters:", ADAPTER_PATH)

# ======================================================
# TOKENIZER
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ======================================================
# LOAD PPO MODEL
# ======================================================
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32
).to(device)

model.pretrained_model = PeftModel.from_pretrained(
    model.pretrained_model,
    ADAPTER_PATH,
    is_trainable=True
)

# ======================================================
# LOAD REFERENCE MODEL (FROZEN)
# ======================================================
ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32
).to(device)

ref_model.pretrained_model = PeftModel.from_pretrained(
    ref_model.pretrained_model,
    ADAPTER_PATH,
    is_trainable=False
)

ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

# ======================================================
# TRAINABLE PARAMS — ONLY LoRA + VALUE HEAD
# ======================================================
for name, p in model.named_parameters():
    if "lora_" in name or "v_head" in name:
        p.requires_grad = True
    else:
        p.requires_grad = False

model.train()  

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable}/{total} ({100*trainable/total:.2f}%)")

model.config.use_cache = False
ref_model.config.use_cache = False

# ======================================================
# DATASET
# ======================================================
print("Loading Spider subset...")
random.seed(0)

TRAIN_DBS = [
    "flight_1","student_assessment","store_1","bike_1","book_2","chinook_1",
    "academic","aircraft","car_1","cinema","club_1","csu_1"
]

dataset = load_dataset("spider", split="train")
dataset = dataset.filter(lambda x: x["db_id"] in TRAIN_DBS)

def valid_example(x):
    return 5 <= len(x["question"].split()) <= 40

dataset = dataset.filter(valid_example)
print("Filtered dataset size:", len(dataset))

def sample_example():
    return dataset[random.randrange(len(dataset))]

total_steps = ROLLOUTS_PER_EPOCH

# ======================================================
# DB UTILITIES
# ======================================================
def get_db_path(db_id):
    return os.path.join(DB_ROOT, db_id, f"{db_id}.sqlite")

def get_db_schema(db_path):
    schema_text = ""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()

        for table in tables:
            table_name = table[0]
            columns = cursor.execute(f"PRAGMA table_info({table_name});").fetchall()
            col_names = [col[1] for col in columns]
            schema_text += f"{table_name}({', '.join(col_names)}) "
        conn.close()
    except:
        pass
    return schema_text

# ======================================================
# PROMPT
# ======================================================
def trim_schema(schema: str, max_chars: int = 1200) -> str:
    if schema is None:
        return ""
    schema = str(schema)
    if len(schema) <= max_chars:
        return schema
    return schema[:max_chars]

def build_prompt(question: str, schema: str, use_schema: bool) -> str:
    if not use_schema:
        return f"### Question:\n{question}\n### SQL:"
    schema = trim_schema(schema, max_chars=MAX_SCHEMA_CHARS)
    return f"### Database Schema:\n{schema}\n### Question:\n{question}\n### SQL:"

# ======================================================
# SQL CONSTRAINED DECODING (Optional, kept for safety)
# ======================================================
SQL_KEYWORDS = [
    "select","from","where","join","inner","left","right","full","outer","on",
    "group","by","order","having","limit","distinct","as","and","or","not","in",
    "is","null","like","between","asc","desc","union","intersect","except"
]
SQL_OPERATORS = ["*", ",", ".", "(", ")", "=", "<", ">", "!", "+", "-", "/", "%", "_"]

def _piece_token_str(tok: str) -> str:
    return tok.lstrip("▁")

def _precompute_always_allowed_token_ids():
    vocab_size = len(tokenizer)
    allowed = set()
    for tid in [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id]:
        if tid is not None and tid >= 0:
            allowed.add(int(tid))
    for s in [" ", "\n", "\t"]:
        allowed.update(tokenizer.encode(s, add_special_tokens=False))
    
    op_chars = set("".join(SQL_OPERATORS))
    for tid in range(vocab_size):
        tok = tokenizer.convert_ids_to_tokens(tid)
        if not isinstance(tok, str) or not tok:
            continue
        piece = _piece_token_str(tok)
        if not piece:
            continue
        if all(ch in op_chars for ch in piece) or piece.isdigit():
            allowed.add(tid)

    for kw in SQL_KEYWORDS:
        for variant in {kw, kw.upper()}:
            allowed.update(tokenizer.encode(" " + variant, add_special_tokens=False))
            allowed.update(tokenizer.encode(variant, add_special_tokens=False))
    return allowed

ALWAYS_ALLOWED_TOKEN_IDS = _precompute_always_allowed_token_ids()

class SQLVocabularyLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = {int(i) for i in allowed_token_ids if int(i) >= 0}
        self._bias = None
        self._bias_vocab_size = None

    def _get_bias(self, scores: torch.Tensor) -> torch.Tensor:
        vocab_size = int(scores.shape[-1])
        if self._bias is None or self._bias_vocab_size != vocab_size:
            bias = torch.full((vocab_size,), float("-inf"), device=scores.device, dtype=scores.dtype)
            for tid in self.allowed_token_ids:
                if tid < vocab_size:
                    bias[tid] = 0.0
            self._bias = bias
            self._bias_vocab_size = vocab_size
        return self._bias

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return scores + self._get_bias(scores)

# ======================================================
# PPO CONFIG (STABLE POLICY LEARNING)
# ======================================================
ppo_config = PPOConfig(
    learning_rate=5e-6,          
    batch_size=8,
    mini_batch_size=2,
    gradient_accumulation_steps=2,
    ppo_epochs=1,
    init_kl_coef=0.2,            
    target_kl=0.02,
    adap_kl_ctrl=True,
    cliprange=0.1,
    cliprange_value=0.1,
    whiten_rewards=False,
    kl_penalty="kl",
    max_grad_norm=0.5,
)

trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

try:
    model.device = torch.device(device)
except Exception:
    pass

# ======================================================
# GENERATION CONFIG
# ======================================================
generation_kwargs = dict(
    max_new_tokens=MAX_OUTPUT_TOKENS,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    top_k=50,
    num_beams=1,
    repetition_penalty=1.05,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# ======================================================
# TRAIN LOOP (FIXED BATCHING)
# ======================================================
print("Starting RL training 🚀 (CodeT5 PPO Stable)")

best_reward = -1e9
model.train()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_reward_sum = 0
    valid_sql_count = 0
    total_seen = 0

    # Initialize batch buffers for TRL
    batch_queries = []
    batch_responses = []
    batch_rewards = []

    for step in range(ROLLOUTS_PER_EPOCH):
        example = dataset[random.randrange(len(dataset))]
        question = example["question"]
        gold_sql = example["query"]
        db_id = example["db_id"]
        db_path = get_db_path(db_id)

        # ---------- PROMPT ----------
        schema = get_db_schema(db_path)
        prompt = build_prompt(question, schema, use_schema=True)

        query_tensor = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).input_ids[0].to(device)

        # ---------- GENERATE ----------
        response_tensors = trainer.generate(
            [query_tensor],
            **generation_kwargs
        )
        response_tensor = response_tensors[0]

        response = tokenizer.decode(response_tensor, skip_special_tokens=True)
        total_seen += 1

        # ---------- BASIC SQL FILTER ----------
        if "select" not in response.lower():
            continue

        # ---------- EXECUTION REWARD ----------
        reward = execution_reward(response, db_path, gold_sql)
        if reward is None:
            continue

        reward = float(reward)

        # ---------- TABLE BONUS ----------
        pred_tables = extract_tables(response)
        gold_tables = extract_tables(gold_sql)
        if len(gold_tables) > 0:
            reward += 0.25 * (len(pred_tables & gold_tables) / len(gold_tables))

        # ---------- COLUMN BONUS ----------
        pred_cols = extract_columns(response)
        gold_cols = extract_columns(gold_sql)
        if len(gold_cols) > 0:
            reward += 0.15 * (len(pred_cols & gold_cols) / len(gold_cols))

        # ---------- CLAMP ----------
        reward = max(-1.0, min(1.0, reward))
        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)

        # ---------- APPEND TO BATCH ----------
        batch_queries.append(query_tensor)
        batch_responses.append(response_tensor)
        batch_rewards.append(reward_tensor)
        
        epoch_reward_sum += reward
        valid_sql_count += 1

        # ---------- PPO UPDATE ----------
        # Only step when our batch is full (matching batch_size=8)
        if len(batch_queries) == ppo_config.batch_size:
            try:
                trainer.step(
                    batch_queries,
                    batch_responses,
                    batch_rewards
                )
            except Exception as e:
                print("⚠️ PPO skipped:", e)
            
            # Clear buffers after stepping
            batch_queries = []
            batch_responses = []
            batch_rewards = []

        # ---------- LOG ----------
        if step % LOG_EVERY == 0 and valid_sql_count > 0:
            print("\n---------------------------")
            print(f"Epoch {epoch}/{NUM_EPOCHS} Step {step}/{ROLLOUTS_PER_EPOCH}")
            print("Avg Reward:", round(epoch_reward_sum/valid_sql_count,3))
            print("Valid SQL:", valid_sql_count,"/",total_seen)
            print("DB:", db_id)
            print("Q:", question)
            print("SQL:", response)
            print("Reward:", round(reward,3))

    # ---------- SAVE BEST MODEL (INSIDE EPOCH) ----------
    avg_reward = epoch_reward_sum / max(valid_sql_count, 1)

    if avg_reward > best_reward:
        best_reward = avg_reward
        save_path = os.path.join(PROJECT_ROOT, "checkpoints/best_rlhf_model")
        os.makedirs(save_path, exist_ok=True)
        
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        print(f"\n✅ Saved BEST RLHF model (reward {best_reward:.3f})")