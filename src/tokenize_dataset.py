from datasets import Dataset
from transformers import T5Tokenizer
import pandas as pd

print("Loading processed dataset...")
train = pd.read_csv("../data/processed/train.csv")
val   = pd.read_csv("../data/processed/validation.csv")

# remove hidden pandas index column if exists
train = train.drop(columns=[c for c in train.columns if "index" in c.lower()], errors="ignore")
val   = val.drop(columns=[c for c in val.columns if "index" in c.lower()], errors="ignore")

print("Loading tokenizer (t5-small)...")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

SQL_PREFIX = "translate English to SQL: "

# ------------------------------------------------------
# TOKENIZATION FUNCTION
# ------------------------------------------------------
def tokenize(example):

    # input = schema + question
    input_text = SQL_PREFIX + example["input"]

    # target = real SQL
    target_sql = example["sql"]

    model_inputs = tokenizer(
        input_text,
        text_target=target_sql,
        max_length=256,
        padding="max_length",
        truncation=True
    )

    return model_inputs


# ------------------------------------------------------
# DATASET CONVERSION
# ------------------------------------------------------
print("Preparing dataset...")
train_ds = Dataset.from_pandas(train)
val_ds   = Dataset.from_pandas(val)

print("Tokenizing train...")
train_ds = train_ds.map(tokenize, remove_columns=train_ds.column_names)

print("Tokenizing validation...")
val_ds = val_ds.map(tokenize, remove_columns=val_ds.column_names)

# save tokenized dataset
train_ds.save_to_disk("../data/tokenized/train")
val_ds.save_to_disk("../data/tokenized/validation")

print("DONE ✔ Tokenized dataset saved correctly")
