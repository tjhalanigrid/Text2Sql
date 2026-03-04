import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model, TaskType

device = "mps" if torch.backends.mps.is_available() else "cpu"

MODEL_PATH = "../outputs/model"   # your supervised trained model

print("Loading base model...")
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)

tokenizer = T5Tokenizer.from_pretrained("t5-small")

# ---------------- LoRA CONFIG ----------------
lora_config = LoraConfig(
    r=8,                       # rank (small brain attachment)
    lora_alpha=16,
    target_modules=["q", "v"], # attention matrices only
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

print("Attaching LoRA adapters...")
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

print("READY ✔ LoRA model loaded")

