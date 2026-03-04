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
MAX_OUTPUT_TOKENS = 64     # 🚀 Speed up: Reduced max tokens
ROLLOUTS_PER_EPOCH = 1024  

# ======================================================
# PATHS
# ======================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs/rlhf_text2sql")
output_dir = RL_MODEL_PATH
DB_ROOT = os.path.join(PROJECT_ROOT, "data/database")

# Explicit resume checkpoint
RESUME_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints/milestone_before_more_dbs")

ADAPTER_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, "checkpoints/sft_adapter_codet5"))
FALLBACK_ADAPTER_PATH = ADAPTER_PATH
FALLBACK_ADAPTER_PATH_2 = os.path.join(PROJECT_ROOT, "checkpoints")

BASE_MODEL = os.environ.get("BASE_MODEL", "Salesforce/codet5-base")

# ======================================================
# LOAD MODEL (LoRA)
# ======================================================
def find_valid_adapter(path_candidates):
    # 🚀 SAFETY & RESUME: Check for existing milestone first
    if os.path.exists(os.path.join(RESUME_CHECKPOINT, "adapter_config.json")):
        print(f"\n✅ Resuming RL training from checkpoint: {RESUME_CHECKPOINT}\n")
        return RESUME_CHECKPOINT
        
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

print("Loading adapter:", ADAPTER_PATH)

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

# 🚀 RESUME: Load adapter dynamically and ensure it's trainable
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
    # already trained
    "flight_1","student_assessment","store_1","bike_1","book_2","chinook_1",
    "academic","aircraft","car_1","cinema","club_1","csu_1",

    # medium difficulty (NEW)
    "college_1","college_2","company_1","company_employee",
    "customer_complaints","department_store","employee_hire_evaluation",
    "museum_visit","products_for_hire","restaurant_1",
    "school_finance","shop_membership","small_bank_1",
    "soccer_1","student_1","tvshow","voter_1","world_1"
]

dataset = load_dataset("spider", split="train")
dataset = dataset.filter(lambda x: x["db_id"] in TRAIN_DBS)

def valid_example(x):
    return 5 <= len(x["question"].split()) <= 40

dataset = dataset.filter(valid_example)
print("Filtered dataset size:", len(dataset))

def sample_example():
    return dataset[random.randrange(len(dataset))]

# ======================================================
# DB UTILITIES
# ======================================================
def get_db_path(db_id):
    return os.path.join(DB_ROOT, db_id, f"{db_id}.sqlite")

# 🚀 SPEED OPTIMIZATION: Cache schema so we don't spam disk IO
_SCHEMA_CACHE = {}

def get_db_schema_cached(db_path):
    if db_path in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[db_path]
        
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
        
    _SCHEMA_CACHE[db_path] = schema_text
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
# 🚀 SPEED OPTIMIZATION: generation limits and randomness bypass
generation_kwargs = dict(
    max_new_tokens=MAX_OUTPUT_TOKENS,
    do_sample=True,          # TRL Requires do_sample=True
    temperature=1.0,         # Disabled randomness logic 
    top_p=1.0,               # Disabled randomness logic
    top_k=0,                 # Disabled randomness logic
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# ======================================================
# TRAIN LOOP (BATCHED & OPTIMIZED)
# ======================================================
print("Starting RL training 🚀 (CodeT5 PPO Stable)")

best_reward = -1e9
global_ppo_step = 0
model.train()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_reward_sum = 0
    valid_sql_count = 0
    total_seen = 0

    # Process in exact chunks matching batch_size to avoid buffer remnants
    for step in range(0, ROLLOUTS_PER_EPOCH, ppo_config.batch_size):
        
        batch_prompts = []
        batch_meta = [] # Store tuple of (question, gold_sql, db_path, db_id)

        # 🚀 BATCH PREPARATION
        for _ in range(ppo_config.batch_size):
            example = sample_example()
            question = example["question"]
            gold_sql = example["query"]
            db_id = example["db_id"]
            db_path = get_db_path(db_id)

            schema = get_db_schema_cached(db_path)
            prompt = build_prompt(question, schema, use_schema=True)
            
            batch_prompts.append(prompt)
            batch_meta.append((question, gold_sql, db_path, db_id))

        # 🚀 SPEED OPTIMIZATION: Padded Batch Tokenization (Multiple of 8)
        encoded_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            pad_to_multiple_of=8
        ).to(device)
        
        # TRL expects lists of 1D tensors
        query_tensors = [encoded_inputs.input_ids[i] for i in range(ppo_config.batch_size)]

        # 🚀 SPEED OPTIMIZATION: Disable gradients for generation pass
        with torch.no_grad():
            response_tensors = trainer.generate(
                query_tensors,
                **generation_kwargs
            )

        batch_rewards = []
        batch_responses_text = []

        # 🚀 BATCH SQL REWARD EXECUTION (Strictly CPU strings)
        for i in range(ppo_config.batch_size):
            response = tokenizer.decode(response_tensors[i], skip_special_tokens=True)
            batch_responses_text.append(response)
            question, gold_sql, db_path, db_id = batch_meta[i]
            
            total_seen += 1

            # ---------- BASIC SQL FILTER ----------
            if "select" not in response.lower():
                batch_rewards.append(torch.tensor(-1.0, dtype=torch.float32).to(device))
                continue

            # ---------- EXECUTION REWARD ----------
            reward = execution_reward(response, db_path, gold_sql)
            if reward is None:
                batch_rewards.append(torch.tensor(-1.0, dtype=torch.float32).to(device))
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
            batch_rewards.append(torch.tensor(reward, dtype=torch.float32).to(device))
            
            epoch_reward_sum += reward
            valid_sql_count += 1

        # ---------- PPO UPDATE ----------
        try:
            trainer.step(
                query_tensors,
                response_tensors,
                batch_rewards
            )
            global_ppo_step += 1
        except Exception as e:
            print("⚠️ PPO skipped:", e)
            continue

        # 🚀 AUTO CHECKPOINT SAVING: Every 200 PPO Updates
        if global_ppo_step > 0 and global_ppo_step % 200 == 0:
            step_save_path = os.path.join(PROJECT_ROOT, f"checkpoints/rl_step_{global_ppo_step}")
            os.makedirs(step_save_path, exist_ok=True)
            
            # Saves ONLY the adapter, keeping disk usage tiny!
            model.save_pretrained(step_save_path)
            tokenizer.save_pretrained(step_save_path)
            print(f"\n💾 [AUTO-SAVE] Checkpoint saved at PPO step {global_ppo_step} -> {step_save_path}")

        # ---------- LOG ----------
        if step % (LOG_EVERY * ppo_config.batch_size) == 0 and valid_sql_count > 0:
            print("\n---------------------------")
            print(f"Epoch {epoch}/{NUM_EPOCHS} Step {step}/{ROLLOUTS_PER_EPOCH} | Global Update {global_ppo_step}")
            print("Avg Reward:", round(epoch_reward_sum/valid_sql_count,3))
            print("Valid SQL:", valid_sql_count,"/",total_seen)
            
            # Print sample from latest batch
            sample_idx = random.randint(0, ppo_config.batch_size - 1)
            print("DB:", batch_meta[sample_idx][3])
            print("Q:", batch_meta[sample_idx][0])
            print("SQL:", batch_responses_text[sample_idx])
            print("Reward:", round(batch_rewards[sample_idx].item(), 3))

    # ---------- SAVE BEST MODEL (INSIDE EPOCH) ----------
    avg_reward = epoch_reward_sum / max(valid_sql_count, 1)

    if avg_reward > best_reward:
        best_reward = avg_reward
        save_path = os.path.join(PROJECT_ROOT, "checkpoints/best_rlhf_model")
        os.makedirs(save_path, exist_ok=True)
        
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        print(f"\n✅ Saved BEST RLHF model for Epoch {epoch} (reward {best_reward:.3f})")