from __future__ import annotations

import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from prompting import clean_gold_sql, get_schema_text, build_prompt

# =====================================================
# SETTINGS
# =====================================================
BASE_MODEL = os.environ.get("BASE_MODEL", "facebook/bart-base")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "sft_best_bart_2")

TRAIN_SPLIT = "train[:7000]"

EPOCHS = 12
LR = 3e-4
PER_DEVICE_BATCH = 16
GRAD_ACCUM = 4

MAX_INPUT = 512
MAX_OUTPUT = 128

# =====================================================
# DEVICE
# =====================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print("Using device:", device)

# =====================================================
# TOKENIZER
# =====================================================
print("Loading tokenizer/model:", BASE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# =====================================================
# PREPROCESS FUNCTION
# =====================================================
def preprocess_function(example):

    question = example["question"]
    db_id = example["db_id"]
    gold_sql = clean_gold_sql(example["query"])

    # ---- Build Prompt ----
    schema_text = get_schema_text(db_id)
    prompt = build_prompt(question, db_id, schema_text=schema_text, training_sql=None)

    model_inputs = tokenizer(
        prompt,
        max_length=MAX_INPUT,
        truncation=True,
        padding="max_length",
    )

    # ---- Target SQL ----
    labels = tokenizer(
        gold_sql,
        max_length=MAX_OUTPUT,
        truncation=True,
        padding="max_length",
    )["input_ids"]

    # IMPORTANT: ignore padding in loss
    labels = [
        (tok if tok != tokenizer.pad_token_id else -400)
        for tok in labels
    ]

    model_inputs["labels"] = labels
    return model_inputs

# =====================================================
# DATASET
# =====================================================
print("Loading Spider subset:", TRAIN_SPLIT)
dataset = load_dataset("spider", split=TRAIN_SPLIT)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

train_ds = dataset["train"]
eval_ds = dataset["test"]

print("Tokenizing dataset (single process, stable)...")

train_tok = train_ds.map(
    preprocess_function,
    batched=False,
    num_proc=1,
    remove_columns=train_ds.column_names,
    load_from_cache_file=False,
)

eval_tok = eval_ds.map(
    preprocess_function,
    batched=False,
    num_proc=1,
    remove_columns=eval_ds.column_names,
    load_from_cache_file=False,
)

print("Train dataset size:", len(train_tok))
print("Eval dataset size:", len(eval_tok))

# =====================================================
# MODEL + LoRA
# =====================================================
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

base_model.config.use_cache = False

# 🚀 UPGRADE 1: Expanded LoRA brainpower
lora_config = LoraConfig(
    r=16,            # Increased rank for more learning capacity
    lora_alpha=32,   # Alpha is typically 2x the rank
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
    # Target all attention and dense layers in BART
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],   
)

model = get_peft_model(base_model, lora_config)
model.to(device)

# =====================================================
# TRAINER
# =====================================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)

args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(PROJECT_ROOT, "checkpoints", "sft_bart_runs"),
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    
    # 🚀 UPGRADE 2 & 3: Better optimization & generalization
    warmup_ratio=0.05,              # Slowly ramp up learning rate
    weight_decay=0.01,              # Penalize over-reliance on single tokens
    label_smoothing_factor=0.1,     # Prevent overconfidence in SQL token matching
    
    evaluation_strategy="epoch",
    save_strategy="epoch",
    
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False, 
    
    logging_steps=50,
    report_to=[],
    fp16=False,
    bf16=False,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# =====================================================
# TRAIN
# =====================================================
trainer.train()

# =====================================================
# SAVE BEST MODEL
# =====================================================
print("Saving best BART LoRA adapter to:", OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

trainer.model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("DONE ✔ SFT BART finished")