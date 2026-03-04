import argparse
import os

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from peft import PeftModel

from prompting import encode_prompt


def main():
    parser = argparse.ArgumentParser(description="Generate SQL from a question + db_id using the RLHF model.")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--db_id", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default=None, help="Defaults to outputs/rlhf_text2sql")
    parser.add_argument("--use_schema", action="store_true", help="Include schema in the prompt (must match training).")
    parser.add_argument("--max_schema_chars", type=int, default=1500)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    adapter_dir = args.model_dir or os.path.join(project_root, "outputs", "rlhf_text2sql")
    base_model = os.environ.get("BASE_MODEL", "t5-small")
    fallback_base_model = os.path.join(project_root, "models", "t5_spider_sft")
    if not os.path.isdir(base_model) and os.path.isdir(fallback_base_model):
        base_model = fallback_base_model

    local_only = not os.path.isdir(base_model)
    tokenizer_source = adapter_dir if os.path.isdir(adapter_dir) else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=not os.path.isdir(tokenizer_source))
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model, local_files_only=local_only).to(device)
    model = PeftModel.from_pretrained(base, adapter_dir).to(device)
    # Merge adapters for faster/stabler generation.
    model = model.merge_and_unload()
    model.config.use_cache = False

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = encode_prompt(
        tokenizer,
        args.question,
        args.db_id,
        device=device,
        max_input_tokens=512,
    )

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        num_beams=1,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        out = model.generate(input_ids=input_ids.unsqueeze(0), **gen_kwargs)

    sql = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    print(sql)


if __name__ == "__main__":
    main()
