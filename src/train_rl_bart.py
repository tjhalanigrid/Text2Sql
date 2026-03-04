# =========================================================
# RLHF TRAINING FOR TEXT2SQL (OPTIMIZED PPO VERSION - BART)
# =========================================================
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
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
# TRAINING SETTINGS (🚀 OPTIMIZED FOR SPEED)
# ======================================================
NUM_EPOCHS = 10         # Increased to compensate for faster epochs
LOG_EVERY = 5              # Print logs much more frequently
MAX_SCHEMA_CHARS = 1500
MAX_OUTPUT_TOKENS = 48     # 🚀 Down from 64. 95% of Spider SQL is <40 tokens.
ROLLOUTS_PER_EPOCH = 256   # 🚀 Down from 1024. Epochs will finish 4x faster!

# ======================================================
# PATHS
# ======================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_ROOT = os.path.join(PROJECT_ROOT, "data/database")

# 🎯 Strict Input: Load strictly from your SFT BART checkpoint
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "checkpoints/sft_best_bart_2")

# 🎯 Strict Output: Save strictly to rl_best_bart
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "checkpoints/rl_best_bart")

BASE_MODEL = os.environ.get("BASE_MODEL", "facebook/bart-base")

if not os.path.exists(ADAPTER_PATH):
    raise RuntimeError(f"❌ No valid LoRA adapter found at: {ADAPTER_PATH}")

print("Loading base:", BASE_MODEL)
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
            schema_text += f"{table_name}({', '.join(col_names)})\n"
        conn.close()
    except:
        pass
        
    _SCHEMA_CACHE[db_path] = schema_text.strip()
    return _SCHEMA_CACHE[db_path]

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

def build_prompt(question: str, schema: str) -> str:
    schema = trim_schema(schema, max_chars=MAX_SCHEMA_CHARS)
    return f"Database Schema:\n{schema}\n\nTranslate English to SQL:\n{question}\nSQL:\n"

# ======================================================
# PPO CONFIG (STABLE POLICY LEARNING)
# ======================================================
ppo_config = PPOConfig(
    learning_rate=3e-6,          # slower = prevents policy jump (very important)
    batch_size=8,
    mini_batch_size=4,           # good size, keep this
    gradient_accumulation_steps=2,

    ppo_epochs=2,                # smoother policy update (was 1 → unstable)

    # ---- KL CONTROL (main fix for negative KL) ----
    init_kl_coef=0.1,
    target_kl=0.08,              # 0.02 was too strict → caused oscillation
    adap_kl_ctrl=True,

    # ---- CLIPPING ----
    cliprange=0.15,
    cliprange_value=0.15,

    # ---- REWARD STABILITY ----
    whiten_rewards=True,         # VERY IMPORTANT for binary execution reward
    kl_penalty="kl",

    # ---- GRADIENT SAFETY ----
    max_grad_norm=0.3,
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
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
# ======================================================
# TRAIN LOOP (BATCHED & OPTIMIZED)
# ======================================================
print("Starting RL training 🚀 (BART PPO Optimized)")

best_reward = -1e9
global_ppo_step = 0
model.train()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_reward_sum = 0
    valid_sql_count = 0
    total_seen = 0

    for step in range(0, ROLLOUTS_PER_EPOCH, ppo_config.batch_size):
        
        batch_prompts = []
        batch_meta = [] 

        for _ in range(ppo_config.batch_size):
            example = sample_example()
            question = example["question"]
            gold_sql = example["query"]
            db_id = example["db_id"]
            db_path = get_db_path(db_id)

            schema = get_db_schema_cached(db_path)
            prompt = build_prompt(question, schema)
            
            batch_prompts.append(prompt)
            batch_meta.append((question, gold_sql, db_path, db_id))

        encoded_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            pad_to_multiple_of=8
        ).to(device)
        
        query_tensors = [encoded_inputs.input_ids[i] for i in range(ppo_config.batch_size)]

        # 🎯 BYPASS: Native model.generate to prevent TRL's truncation crash
        with torch.no_grad():
            response_tensors_raw = model.generate(
                input_ids=encoded_inputs.input_ids,
                attention_mask=encoded_inputs.attention_mask,
                **generation_kwargs
            )

        batch_rewards = []
        batch_responses_text = []
        response_tensors = []

        for i in range(ppo_config.batch_size):
            resp = response_tensors_raw[i]
            
            # 🎯 Strip padding safely so TRL's mask calculation never crashes
            non_pad_mask = resp != tokenizer.pad_token_id
            if non_pad_mask.sum() == 0:
                resp = torch.tensor([tokenizer.eos_token_id], device=device)
                non_pad_mask = resp != tokenizer.pad_token_id
                
            valid_len = non_pad_mask.nonzero()[-1].item() + 1
            clean_resp = resp[:valid_len]
            response_tensors.append(clean_resp)

            response = tokenizer.decode(clean_resp, skip_special_tokens=True)
            batch_responses_text.append(response)
            
            question, gold_sql, db_path, db_id = batch_meta[i]
            total_seen += 1

            if "select" not in response.lower():
                batch_rewards.append(torch.tensor(-1.0, dtype=torch.float32).to(device))
                continue

            reward = execution_reward(response, db_path, gold_sql)
            if reward is None:
                batch_rewards.append(torch.tensor(-1.0, dtype=torch.float32).to(device))
                continue

            reward = float(reward)

            pred_tables = extract_tables(response)
            gold_tables = extract_tables(gold_sql)
            if len(gold_tables) > 0:
                reward += 0.25 * (len(pred_tables & gold_tables) / len(gold_tables))

            pred_cols = extract_columns(response)
            gold_cols = extract_columns(gold_sql)
            if len(gold_cols) > 0:
                reward += 0.15 * (len(pred_cols & gold_cols) / len(gold_cols))

            reward = max(-1.0, min(1.0, reward))
            batch_rewards.append(torch.tensor(reward, dtype=torch.float32).to(device))
            
            epoch_reward_sum += reward
            valid_sql_count += 1

        # ---------- PPO UPDATE ----------
        try:
            trainer.step(query_tensors, response_tensors, batch_rewards)
            global_ppo_step += 1
        except Exception as e:
            print("⚠️ PPO skipped:", e)
            continue

        # ---------- LOG ----------
        if step % (LOG_EVERY * ppo_config.batch_size) == 0 and valid_sql_count > 0:
            print("\n---------------------------")
            print(f"Epoch {epoch}/{NUM_EPOCHS} Step {step}/{ROLLOUTS_PER_EPOCH} | Global Update {global_ppo_step}")
            print("Avg Reward:", round(epoch_reward_sum/valid_sql_count,3))
            print("Valid SQL:", valid_sql_count,"/",total_seen)
            
            sample_idx = random.randint(0, ppo_config.batch_size - 1)
            print("DB:", batch_meta[sample_idx][3])
            print("Q:", batch_meta[sample_idx][0])
            print("SQL:", batch_responses_text[sample_idx])
            print("Reward:", round(batch_rewards[sample_idx].item(), 3))

    # ---------- SAVE ONLY THE BEST MODEL ----------
    avg_reward = epoch_reward_sum / max(valid_sql_count, 1)

    if avg_reward > best_reward:
        best_reward = avg_reward
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        print(f"\n✅ Saved BEST RLHF model for Epoch {epoch} (reward {best_reward:.3f}) at {OUTPUT_DIR}")