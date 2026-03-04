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
BASE_MODEL = os.environ.get("BASE_MODEL", "t5-small")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 🎯 FIXED: Save final model to checkpoints/sft_t5 to protect existing models
OUT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "sft_t5")

TRAIN_SPLIT = "train[:7000]"
EPOCHS = 8
LR = 3e-4
PER_DEVICE_BATCH = 4
GRAD_ACCUM = 2

MAX_INPUT = 512
MAX_OUTPUT = 128

# =====================================================
# DEVICE
# =====================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# TOKENIZER
# =====================================================
print("Loading tokenizer/model:", BASE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# =====================================================
# PREPROCESS FUNCTION (CRITICAL FIXED VERSION)
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
        (tok if tok != tokenizer.pad_token_id else -100)
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
    num_proc=1,                    # 🔥 VERY IMPORTANT FIX
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
base_model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
    target_modules=["q", "v"],   # correct for T5
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
    # 🎯 FIXED: Changed path to prevent mixing logs with your old CodeT5 logs
    output_dir=os.path.join(PROJECT_ROOT, "checkpoints", "sft_t5_runs"),
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    evaluation_strategy="epoch",
    
    # 🎯 FIXED: "no" completely stops intermediate saving! Only the final model will be saved.
    save_strategy="no", 
    
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
# SAVE
# =====================================================
print("Saving LoRA adapter to:", OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("DONE ✔ SFT warmup finished")