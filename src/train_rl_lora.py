# ======================================
# RLHF Text2SQL — FINAL WORKING VERSION
# T5-small + LoRA + PPO + Execution Reward
# Single-sample stable training (Mac MPS safe)
# ======================================

from execution_reward import execution_reward
import os, gc, json, random, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from trl import PPOTrainer, PPOConfig
from trl.models.modeling_value_head import AutoModelForSeq2SeqLMWithValueHead
from peft import LoraConfig, get_peft_model

# ---------------- SETTINGS ----------------
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

os.makedirs("rlhf_text2sql_lora", exist_ok=True)

# ---------------- MODEL ----------------
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q","v"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)

base_model = get_peft_model(base_model, lora_config)

model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(base_model).to(device)
ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_name).to(device)

model.config.use_cache = False
ref_model.config.use_cache = False

# ---------------- DATA ----------------
with open("data/train_spider.json") as f:
    dataset = json.load(f)

def build_prompt(example):
    return f"Translate to SQL: {example['question']}"

# ---------------- PPO ----------------
ppo_config = PPOConfig(
    batch_size=1,
    mini_batch_size=1,
    learning_rate=2e-6,
    target_kl=0.05,
    adap_kl_ctrl=True,
    init_kl_coef=0.2,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# ---------------- GENERATION ----------------
def generate_sql(query_tensors):

    # deterministic decoding = prevents NaN explosion
    with torch.no_grad():
        response_tensors = ppo_trainer.generate(
            query_tensors,
            max_new_tokens=64,

            # 🔴 CRITICAL: disable sampling
            do_sample=False,

            # stable decoding
            num_beams=1,
            early_stopping=True,

            # prevents invalid tokens
            pad_token_id=tokenizer.eos_token_id,
        )

    # extra safety (important on MPS)
    cleaned = []
    for t in response_tensors:
        t = torch.nan_to_num(t, nan=0, posinf=0, neginf=0)
        cleaned.append(t)

    return cleaned

# ---------------- TRAIN ----------------
MAX_STEPS = 1200

for step in range(MAX_STEPS):

    # pick random Spider example
    example = random.choice(dataset)

    question = example["question"]
    gold_sql = example["query"]
    db_id = example["db_id"]
    db_path = f"data/database/{db_id}/{db_id}.sqlite"

    # tokenize
    enc = tokenizer(build_prompt(example), return_tensors="pt")
    query_tensor = enc.input_ids.to(device)
    query_tensors = [query_tensor[0]]

    # generate SQL
    response_tensors = generate_sql(query_tensors)
    pred_sql = tokenizer.decode(response_tensors[0], skip_special_tokens=True)

    # -------- EXECUTION REWARD --------
    reward = execution_reward(pred_sql, gold_sql, db_path)
    reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)

    # PPO update
    stats = ppo_trainer.step(query_tensors, response_tensors, [reward_tensor])

    # stabilize
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # cleanup
    del query_tensor, response_tensors, reward_tensor
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # log
    if step % 20 == 0:
        print(f"\nStep {step}/{MAX_STEPS}")
        print("DB:", db_id)
        print("Q:", question)
        print("Pred:", pred_sql)
        print("Gold:", gold_sql)
        print("Reward:", reward)

# ---------------- SAVE ----------------
model.save_pretrained("rlhf_text2sql_lora")
tokenizer.save_pretrained("rlhf_text2sql_lora")

print("\nTraining complete — model saved!")