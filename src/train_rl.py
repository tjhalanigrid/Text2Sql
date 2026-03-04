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
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ======================================================
# TRAINING SETTINGS
# ======================================================
NUM_EPOCHS = 5
LOG_EVERY = 20
USE_SCHEMA = True
SCHEMA_WARMUP_EPOCHS = 0
MAX_SCHEMA_CHARS = 1500
MAX_OUTPUT_TOKENS = 80
ROLLOUTS_PER_EPOCH = 2048


# ======================================================
# PATHS
# ======================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 🎯 FIXED: Save ONLY the best model to this exact path
RL_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "rlhf_t5_best")
output_dir = RL_MODEL_PATH

DB_ROOT = os.path.join(PROJECT_ROOT, "data/database")

# 🎯 Updated to point to our newly trained t5-small SFT model
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "checkpoints/sft_t5") 

FALLBACK_ADAPTER_PATH = os.path.join(PROJECT_ROOT, "models/t5_spider_sft_lora")
FALLBACK_ADAPTER_PATH_2 = os.path.join(PROJECT_ROOT, "outputs/sft_text2sql")
# 🎯 ENSURING t5-small is used
BASE_MODEL = os.environ.get("BASE_MODEL", "t5-small")


# ======================================================
# LOAD MODEL (LoRA)
# ======================================================
print("Loading base:", BASE_MODEL)
if not os.path.isdir(ADAPTER_PATH):
    if os.path.isdir(FALLBACK_ADAPTER_PATH):
        ADAPTER_PATH = FALLBACK_ADAPTER_PATH
    elif os.path.isdir(FALLBACK_ADAPTER_PATH_2):
        ADAPTER_PATH = FALLBACK_ADAPTER_PATH_2
print("Loading adapters:", ADAPTER_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(BASE_MODEL).to(device)
model.pretrained_model = PeftModel.from_pretrained(model.pretrained_model, ADAPTER_PATH)

ref_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(BASE_MODEL).to(device)
ref_model.pretrained_model = PeftModel.from_pretrained(ref_model.pretrained_model, ADAPTER_PATH)

ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad_(False)

# Freeze base transformer weights; train LoRA adapters + value head.
for name, p in model.named_parameters():
    # Train value head
    if name.startswith("v_head"):
        p.requires_grad = True
    # Train LoRA adapters (policy learning!)
    elif "lora_" in name:
        p.requires_grad = True
    # Freeze base model
    else:
        p.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable}/{total} ({100*trainable/total:.2f}%)")

model.config.use_cache = False
ref_model.config.use_cache = False

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


# ======================================================
# DATASET
# ======================================================
print("Loading Spider subset...")
random.seed(0)

# Train on a small, stable curriculum of DBs first.
TRAIN_DBS = [
    "flight_1",
    "student_assessment",
    "store_1",
    "bike_1",
    "book_2",
    "chinook_1",
]

dataset = load_dataset("spider", split="train")
_TRAIN_DBS_SET = set(TRAIN_DBS)
dataset = dataset.filter(lambda x: x["db_id"] in _TRAIN_DBS_SET)
dataset = dataset.select(range(min(800, len(dataset))))

print("Using RLHF DBs:", TRAIN_DBS)
print("Filtered size:", len(dataset))

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

        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()

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
PREFIX = "translate English to SQL:"


def trim_schema(schema: str, max_chars: int = 1200) -> str:
    if schema is None:
        return ""
    schema = str(schema)
    if len(schema) <= max_chars:
        return schema
    return schema[:max_chars]
def build_prompt(question: str, schema: str, use_schema: bool) -> str:
    if not use_schema:
        return f"{PREFIX}\n\nQuestion:\n{question}\n\nSQL:"

    schema = trim_schema(schema, max_chars=MAX_SCHEMA_CHARS)
    return f"{PREFIX}\n\nSchema:\n{schema}\n\nQuestion:\n{question}\n\nSQL:"

def encode_prompt(question, schema, use_schema):
    # Never truncate the question; only truncate schema tokens if needed.
    if not use_schema:
        prompt = build_prompt(question, schema, use_schema=False)
        return tokenizer(prompt, return_tensors="pt", truncation=True).input_ids[0].to(device)

    schema = trim_schema(schema, max_chars=MAX_SCHEMA_CHARS)
    prefix_schema = f"{PREFIX}\n\nSchema:\n"
    mid = "\n\nQuestion:\n"
    suffix = f"{question}\n\nSQL:"

    prefix_ids = tokenizer.encode(prefix_schema, add_special_tokens=False)
    schema_ids = tokenizer.encode(schema, add_special_tokens=False)
    mid_ids = tokenizer.encode(mid, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)

    max_len = getattr(tokenizer, "model_max_length", 512)
    eos_id = tokenizer.eos_token_id
    max_without_eos = max_len - (1 if eos_id is not None else 0)

    # Ensure the question+SQL suffix always fits; truncate schema first.
    fixed_len = len(prefix_ids) + len(mid_ids) + len(suffix_ids)
    if fixed_len > max_without_eos:
        # Extremely rare; clip the suffix (question) only if unavoidable.
        keep = max(0, max_without_eos - (len(prefix_ids) + len(mid_ids)))
        suffix_ids = suffix_ids[:keep]
        fixed_len = len(prefix_ids) + len(mid_ids) + len(suffix_ids)

    remaining_for_schema = max_without_eos - fixed_len
    if remaining_for_schema < 0:
        remaining_for_schema = 0
    schema_ids = schema_ids[:remaining_for_schema]

    ids = prefix_ids + schema_ids + mid_ids + suffix_ids
    ids = ids[:max_without_eos]
    if eos_id is not None:
        ids = ids + [eos_id]

    return torch.tensor(ids, dtype=torch.long).to(device)


# ======================================================
# SQL CONSTRAINED DECODING
# ======================================================
SQL_KEYWORDS = [
    "select", "from", "where", "join", "inner", "left", "right",
    "full", "outer", "on", "group", "by", "order", "having",
    "limit", "distinct", "as", "and", "or", "not", "in", "is",
    "null", "like", "between", "asc", "desc", "union",
    "intersect", "except",
]

SQL_OPERATORS = ["*", ",", ".", "(", ")", "=", "<", ">", "!", "+", "-", "/", "%", "_"]


def _piece_token_str(tok: str) -> str:
    # T5 SentencePiece: "▁" marks a leading space; strip it for char checks.
    return tok.lstrip("▁")


def _precompute_always_allowed_token_ids():
    vocab_size = len(tokenizer)
    allowed = set()

    # Always allow special tokens.
    for tid in [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id]:
        if tid is not None and tid >= 0:
            allowed.add(int(tid))

    # Allow whitespace/newlines in case they exist as pieces.
    for s in [" ", "\n", "\t"]:
        allowed.update(tokenizer.encode(s, add_special_tokens=False))

    # Allow operators/punctuation/numeric pieces broadly.
    op_chars = set("".join(SQL_OPERATORS))
    for tid in range(vocab_size):
        tok = tokenizer.convert_ids_to_tokens(tid)
        if not isinstance(tok, str) or not tok:
            continue
        piece = _piece_token_str(tok)
        if not piece:
            continue
        if all((ch in op_chars) for ch in piece):
            allowed.add(tid)
            continue
        if piece.isdigit():
            allowed.add(tid)
            continue
        # Common numeric fragments like "1", "00", etc.
        if all(ch.isdigit() for ch in piece):
            allowed.add(tid)

    # Allow keyword pieces.
    for kw in SQL_KEYWORDS:
        for variant in {kw, kw.upper(), kw.capitalize()}:
            allowed.update(tokenizer.encode(" " + variant, add_special_tokens=False))
            allowed.update(tokenizer.encode(variant, add_special_tokens=False))

    return allowed


ALWAYS_ALLOWED_TOKEN_IDS = _precompute_always_allowed_token_ids()


def _schema_allowed_token_ids(table_names, column_names):
    allowed = set(ALWAYS_ALLOWED_TOKEN_IDS)

    def _add_identifier(name: str):
        if not name:
            return
        # Add whole identifier and common splits.
        variants = {name, name.lower(), name.upper()}
        parts = re.split(r"[_\s]+", name)
        variants.update({p for p in parts if p})
        for v in variants:
            allowed.update(tokenizer.encode(" " + v, add_special_tokens=False))
            allowed.update(tokenizer.encode(v, add_special_tokens=False))

    for t in table_names:
        _add_identifier(t)
    for c in column_names:
        _add_identifier(c)

    return allowed


class SQLVocabularyLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = {int(i) for i in allowed_token_ids if int(i) >= 0}
        self._bias = None
        self._bias_vocab_size = None

    def _get_bias(self, scores: torch.Tensor) -> torch.Tensor:
        vocab_size = int(scores.shape[-1])
        if (
            self._bias is None
            or self._bias.device != scores.device
            or self._bias.dtype != scores.dtype
            or self._bias_vocab_size != vocab_size
        ):
            bias = torch.full((vocab_size,), float("-inf"), device=scores.device, dtype=scores.dtype)
            for tid in self.allowed_token_ids:
                if tid < vocab_size:
                    bias[tid] = 0.0
            self._bias = bias
            self._bias_vocab_size = vocab_size
        return self._bias

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        return scores + self._get_bias(scores)


_DB_VOCAB_CACHE = {}


def get_db_tables_columns(db_path: str):
    if db_path in _DB_VOCAB_CACHE:
        return _DB_VOCAB_CACHE[db_path]
    tables, cols = [], []
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        for (tname,) in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        ).fetchall():
            if not tname:
                continue
            tables.append(tname)
            try:
                for row in cur.execute(f'PRAGMA table_info("{tname}")').fetchall():
                    if row and isinstance(row[1], str):
                        cols.append(row[1])
            except Exception:
                continue
        conn.close()
    except Exception:
        pass
    _DB_VOCAB_CACHE[db_path] = (tables, cols)
    return tables, cols


# ======================================================
# PPO CONFIG (stable learning)
# ======================================================
ppo_config = PPOConfig(
    learning_rate=2e-5,            # was 1e-6 → model could not move
    batch_size=8,                  # better gradient estimate
    mini_batch_size=2,
    gradient_accumulation_steps=2, # stable updates on small data
    ppo_epochs=1,

    # --- KL control (MOST IMPORTANT FIX) ---
    init_kl_coef=0.05,             # reduce punishment
    target_kl=0.15,                # relax constraint to avoid skipped updates
    adap_kl_ctrl=True,

    # --- stability ---
    cliprange=0.25,
    cliprange_value=0.25,
    whiten_rewards=True,
    kl_penalty="kl",
    max_grad_norm=1.0,
)
trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)
optimizer = trainer.optimizer

# Provide `.device` attribute for the supervised anchor helper.
try:
    model.device = torch.device(device)
except Exception:
    pass


# ======================================================
# GENERATION (schema-constrained decoding)
# ======================================================
generation_kwargs = dict(
    max_new_tokens=64,         # 128 causes garbage SQL loops

    do_sample=True,
    temperature=0.9,           # encourage exploration
    top_p=0.95,
    top_k=100,

    repetition_penalty=1.1,    # prevents SELECT SELECT SELECT
    no_repeat_ngram_size=3,

    num_beams=1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
# ======================================================
# TRAIN LOOP
# ======================================================
print("Starting RL training 🚀")

query_buffer, response_buffer, reward_buffer, gold_buffer = [], [], [], []
query_text_buffer = []

best_reward = -999999
best_epoch = -1

def _is_parsable_sql(sql: str) -> bool:
    s = (sql or "").strip()
    if not s:
        return False
    up = s.upper()
    if "SELECT" not in up or "FROM" not in up:
        return False
    if sqlparse is None:
        return True
    try:
        return bool(sqlparse.parse(s))
    except Exception:
        return False


def _pad_2d(seqs, pad_id: int):
    max_len = max(int(s.numel()) for s in seqs)
    out = torch.full((len(seqs), max_len), int(pad_id), dtype=torch.long, device=device)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        n = int(s.numel())
        out[i, :n] = s.to(device)
        attn[i, :n] = 1
    return out, attn


def _shift_right(labels: torch.Tensor, start_id: int) -> torch.Tensor:
    dec = labels.clone()
    dec[:, 1:] = labels[:, :-1]
    dec[:, 0] = int(start_id)
    return dec


def safe_get_kl(stats):
    if not isinstance(stats, dict):
        return None
    for k in stats.keys():
        if "kl" in str(k).lower():
            v = stats[k]
            try:
                return float(v.item() if hasattr(v, "item") else v)
            except Exception:
                return None
    return None

def supervised_anchor_step(model, tokenizer, queries, gold_sqls, weight=0.05):
    model.train()
    total_loss = 0.0

    for q, gold in zip(queries, gold_sqls):

        enc = tokenizer(q, return_tensors="pt", truncation=True).to(model.device)
        dec = tokenizer(text_target=gold, return_tensors="pt", truncation=True)

        labels = dec.input_ids.to(model.device)

        # teacher forcing shift
        decoder_input_ids = labels[:, :-1].contiguous()
        target_ids = labels[:, 1:].contiguous()

        outputs = model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            decoder_input_ids=decoder_input_ids,
        )

        logits = outputs[0]

        vocab_size = logits.size(-1)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            target_ids.view(-1),
            ignore_index=tokenizer.pad_token_id,
        )

        (loss * weight).backward()
        total_loss += loss.item()

    return total_loss


@torch.no_grad()
def _estimate_policy_entropy(query_tensors, response_tensors) -> torch.Tensor:
    """
    Returns per-sample average token entropy of the policy on the sampled response tokens.
    Used as a small bonus to reduce repetition collapse.
    """
    pad_id = int(tokenizer.pad_token_id)
    enc_ids, enc_attn = _pad_2d(query_tensors, pad_id)
    dec_ids, dec_attn = _pad_2d(response_tensors, pad_id)

    start_id = int(getattr(model.pretrained_model.config, "decoder_start_token_id", pad_id))
    dec_inp = _shift_right(dec_ids, start_id)

    out = model.pretrained_model(
        input_ids=enc_ids,
        attention_mask=enc_attn,
        decoder_input_ids=dec_inp,
        use_cache=False,
    )
    logp = torch.log_softmax(out.logits, dim=-1)
    p = torch.exp(logp)
    ent = -(p * logp).sum(dim=-1)  # [B, T]
    # average only over non-pad positions of the sampled response
    denom = dec_attn.sum(dim=-1).clamp_min(1)
    return (ent * dec_attn).sum(dim=-1) / denom  # [B]


def _repeat_penalty(response_tensor: torch.Tensor) -> float:
    """
    Penalize repetition to avoid 'SELECT SELECT SELECT' collapse.
    Simple heuristic: consecutive duplicate token ratio + low-unique-token ratio.
    """
    ids = response_tensor.detach().tolist()
    n = len(ids)
    if n <= 1:
        return 0.0
    consec_dup = 0
    for i in range(1, n):
        if ids[i] == ids[i - 1]:
            consec_dup += 1
    unique_ratio = len(set(ids)) / n
    consec_ratio = consec_dup / (n - 1)
    # Higher penalty when low unique + high consecutive duplicates
    return float(0.5 * consec_ratio + 0.5 * (1.0 - unique_ratio))


def _supervised_anchor_step(query_tensors, gold_sql_texts, weight: float = 0.05) -> None:
    """
    Small teacher-forcing step on gold SQL to anchor grammar during PPO.
    Runs only if PPOTrainer exposes (accelerator, optimizer).
    """
    if not gold_sql_texts:
        return
    accelerator = getattr(trainer, "accelerator", None)
    optimizer = getattr(trainer, "optimizer", None)
    if accelerator is None or optimizer is None:
        return

    pad_id = int(tokenizer.pad_token_id)
    enc_ids, enc_attn = _pad_2d(query_tensors, pad_id)

    # Tokenize gold SQL targets (decoder side)
    gold_ids = []
    for s in gold_sql_texts:
        g = (s or "").strip()
        if not g:
            g = "SELECT 1"
        ids = tokenizer.encode(g, add_special_tokens=False)[:256]
        if tokenizer.eos_token_id is not None:
            ids = ids + [int(tokenizer.eos_token_id)]
        gold_ids.append(torch.tensor(ids, dtype=torch.long))

    dec_ids, dec_attn = _pad_2d(gold_ids, pad_id)
    labels = dec_ids.clone()
    labels[dec_attn == 0] = -100

    # PEFT model forward supports labels -> returns loss
    out = model.pretrained_model(
        input_ids=enc_ids,
        attention_mask=enc_attn,
        labels=labels,
        use_cache=False,
    )
    loss = out.loss * float(weight)

    optimizer.zero_grad(set_to_none=True) if hasattr(optimizer, "zero_grad") else None
    accelerator.backward(loss)
    optimizer.step()


def _curriculum_allows(gold_sql: str, epoch_num: int) -> bool:
    gold_up = (gold_sql or "").upper()
    has_join = "JOIN" in gold_up
    has_set_op = any(op in gold_up for op in ["UNION", "INTERSECT", "EXCEPT"])
    tables = extract_tables(gold_sql)
    single_table = len(tables) <= 1 and (not has_join)

    # Epoch 1: only single-table, no joins/set-ops.
    if epoch_num == 1:
        return single_table and (not has_set_op)
    # Epoch 2: allow joins, but still avoid set-ops.
    if epoch_num == 2:
        return (single_table or has_join) and (not has_set_op)
    # Epoch 3+: full dataset.
    return True


for epoch in range(1, NUM_EPOCHS + 1):

    use_schema_this_epoch = USE_SCHEMA and (epoch > SCHEMA_WARMUP_EPOCHS)

    epoch_reward_sum = 0
    negative_rewards = 0
    partial_rewards = 0
    correct_rewards = 0

    total_considered = 0
    valid_sql_count = 0
    exec_correct_count = 0
    table_overlap_sum = 0.0
    column_overlap_sum = 0.0
    kl_values = []

    for step in range(1, total_steps + 1):

        example = dataset[random.randrange(len(dataset))]

        question = example["question"]
        gold_sql = example["query"]
        db_id = example["db_id"]
        db_path = get_db_path(db_id)

        # NOTE: sampling-with-replacement provides more rollouts per epoch.

        schema = get_db_schema(db_path)
        question_text = build_prompt(question, schema, use_schema_this_epoch)
        query_tensor = encode_prompt(question, schema, use_schema_this_epoch)

        # ----- generate -----
        table_names, column_names = get_db_tables_columns(db_path)
        allowed_ids = _schema_allowed_token_ids(table_names, column_names)
        logits_processor = LogitsProcessorList([SQLVocabularyLogitsProcessor(allowed_ids)])

        response = trainer.generate([query_tensor], logits_processor=logits_processor, **generation_kwargs)[0]
        response_tensor = response.squeeze(0)[:MAX_OUTPUT_TOKENS]

        pred_sql = tokenizer.decode(response_tensor.cpu(), skip_special_tokens=True)

        total_considered += 1

        # PPO must optimize ONLY when SQL parses successfully.
        if not _is_parsable_sql(pred_sql):
            negative_rewards += 1
            continue

        # Reject generations shorter than 6 tokens.
        if int(response_tensor.numel()) < 6:
            negative_rewards += 1
            continue

        # ----- reward -----
        reward_value = execution_reward(pred_sql, db_path, gold_sql)

        # SQL validity gate: if invalid/unparsable -> reward_value is None -> skip PPO entirely.
        if reward_value is None:
            if step % 100 == 0:
                ratio = valid_sql_count / max(total_considered, 1)
                print(f"\nLearning ratio: {valid_sql_count}/{total_considered} ({ratio:.3f})")
                if ratio < 0.15:
                    print("MODEL COLLAPSING")
            continue

        # Clip rewards to [-1, 1]
        reward_value = float(max(-1.0, min(1.0, reward_value)))
        # Penalize repetition in decoded output (token-level heuristic).
        reward_value = float(max(-1.0, min(1.0, reward_value - 0.2 * _repeat_penalty(response_tensor))))
        # Keep rewards on CPU for normalization; move to device only for trainer.step().
        reward_tensor = torch.tensor(reward_value, dtype=torch.float32)

        epoch_reward_sum += reward_value

        # ----- metrics -----
        # "Valid sample" means reward is not None (parsable SQL).
        valid_sql_count += 1

        pred_tables = extract_tables(pred_sql)
        gold_tables = extract_tables(gold_sql)
        pred_cols = extract_columns(pred_sql)
        gold_cols = extract_columns(gold_sql)

        if len(gold_tables) > 0:
            table_overlap_sum += len(pred_tables & gold_tables) / max(len(gold_tables), 1)
        if len(gold_cols) > 0:
            column_overlap_sum += len(pred_cols & gold_cols) / max(len(gold_cols), 1)

        # execution_reward returns 1.0 for correct execution result.
        if reward_value >= 1.0:
            exec_correct_count += 1

        if reward_value <= -1.0:
            negative_rewards += 1
        elif reward_value >= 1.0:
            correct_rewards += 1
        else:
            partial_rewards += 1

        # Train only on informative samples:
        # - invalid SQL already skipped (reward is None)
        # - very small magnitude signal skipped
        if abs(reward_value) < 0.1:
            continue

        query_buffer.append(query_tensor)
        response_buffer.append(response_tensor)
        reward_buffer.append(reward_tensor)
        gold_buffer.append(gold_sql)
        query_text_buffer.append(question_text)

        # ----- PPO update -----
        if len(query_buffer) == ppo_config.batch_size:
            # move rewards to device
            reward_buffer = [r.to(device) for r in reward_buffer]

            # run PPO step
            stats = trainer.step(query_buffer, response_buffer, reward_buffer)

            # log KL safely (no control logic)
            kl = safe_get_kl(stats)
            if kl is not None:
                kl_values.append(kl)

            # --- supervised anchor to prevent grammar collapse ---
            supervised_anchor_step(model, tokenizer, query_text_buffer, gold_buffer, weight=0.05)
            optimizer.step()
            optimizer.zero_grad()

            # reset buffers
            query_buffer, response_buffer, reward_buffer, gold_buffer = [], [], [], []
            query_text_buffer = []

        # ----- learning ratio logging -----
        if step % 100 == 0:
            ratio = valid_sql_count / max(total_considered, 1)
            print(f"\nLearning ratio: {valid_sql_count}/{total_considered} ({ratio:.3f})")
            if ratio < 0.15:
                print("MODEL COLLAPSING")
                # Increase KL coefficient dynamically when valid_sql_rate drops.
                try:
                    if hasattr(trainer, "kl_ctl") and hasattr(trainer.kl_ctl, "value"):
                        trainer.kl_ctl.value *= 1.5
                        print(f"Increasing KL coef -> {trainer.kl_ctl.value:.4f}")
                except Exception:
                    pass

        # ----- logging -----
        if step % LOG_EVERY == 0:
            avg_reward = epoch_reward_sum / step
            print("\n---------------------------")
            print(f"Epoch {epoch}/{NUM_EPOCHS} | Step {step}/{total_steps} | Avg Reward {avg_reward:.3f}")
            print("DB:", db_id)
            print("Q:", question)
            print("SQL:", pred_sql)
            print("Reward:", reward_value)

    # epoch stats
    print(f"\nEpoch {epoch} stats:")
    print("negative:", negative_rewards)
    print("partial:", partial_rewards)
    print("correct:", correct_rewards)

    denom = max(total_considered, 1)
    print("\nEpoch metrics:")
    print(f"execution_accuracy: {exec_correct_count/denom:.3f}")
    print(f"valid_sql_rate: {valid_sql_count/denom:.3f}")
    print(f"table_match_rate: {table_overlap_sum/denom:.3f}")
    print(f"column_match_rate: {column_overlap_sum/denom:.3f}")
    print(f"avg_reward: {epoch_reward_sum/max(denom,1):.3f}")
    if kl_values:
        avg_kl = sum(kl_values) / max(len(kl_values), 1)
        print(f"avg_kl: {avg_kl:.3f}")
        if avg_kl < -8:
            print("\nKL collapse guard triggered (avg_kl < -8). Stopping early.")
            break

    # 🎯 FIXED: Removed the code that saved intermediate checkpoints at the end of each epoch

    # Only save if this epoch is the best one so far
    epoch_avg_reward = epoch_reward_sum / max(denom, 1)
    if epoch_avg_reward > best_reward:
        best_reward = epoch_avg_reward
        best_epoch = epoch

        print(f"\nNew best model at epoch {epoch} with reward {best_reward:.4f}")

        # 🎯 FIXED: Save directly to checkpoints/rlhf_t5_best, overwriting if needed
        os.makedirs(output_dir, exist_ok=True)

        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


print(f"\nTraining finished.")
print(f"Best epoch: {best_epoch}")
print(f"Best reward: {best_reward:.4f}")
print(f"Best model saved at: {output_dir}")