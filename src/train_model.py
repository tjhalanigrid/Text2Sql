import torch
from datasets import load_from_disk
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

# ======================================================
# DEVICE (Mac M1/M2/M3 Safe)
# ======================================================
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# ======================================================
# LOAD TOKENIZED DATASET (FIXED PATHS)
# ======================================================
print("Loading tokenized dataset...")

train_dataset = load_from_disk("data/tokenized/train")
val_dataset   = load_from_disk("data/tokenized/validation")

print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))

# ======================================================
# LOAD MODEL
# ======================================================
print("Loading model (t5-small)...")

model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Prevent Mac memory crash
model.config.use_cache = False

# Important T5 settings (prevents generation bugs)
model.config.decoder_start_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# ======================================================
# DATA COLLATOR
# ======================================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# ======================================================
# TRAINING ARGUMENTS (Mac Safe)
# ======================================================
print("Setting training config...")

training_args = Seq2SeqTrainingArguments(
    output_dir="outputs/model",

    evaluation_strategy="epoch",
    save_strategy="epoch",

    learning_rate=3e-4,
    num_train_epochs=5,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,

    logging_steps=50,

    fp16=False,
    bf16=False,
    dataloader_pin_memory=False,

    predict_with_generate=True,
    report_to="none"
)

# ======================================================
# TRAINER
# ======================================================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ======================================================
# TRAIN
# ======================================================
print("Training started 🚀")
trainer.train()

# ======================================================
# SAVE MODEL
# ======================================================
print("Saving model...")
trainer.save_model("outputs/model")
tokenizer.save_pretrained("outputs/model")

print("\nDONE ✔ Base model trained successfully")