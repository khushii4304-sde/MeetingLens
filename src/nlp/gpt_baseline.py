# src/nlp/gpt_baseline.py
# Zero-shot GPT-style baseline using Flan-T5

import sys
import os

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

import torch
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("=" * 60, flush=True)
print("Flan-T5 Zero-Shot Baseline", flush=True)
print("=" * 60, flush=True)

# ── 1. Device detection ──────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}", flush=True)
else:
    device = torch.device("cpu")
    print("⚠ No GPU found — running on CPU (will be slower)", flush=True)

# ── 2. Load model ────────────────────────────────────────────
MODEL_NAME = "google/flan-t5-base"
print(f"\nLoading {MODEL_NAME} ...", flush=True)
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print("✓ Model loaded successfully", flush=True)
except Exception as e:
    print(f"✗ Failed to load model: {e}", flush=True)
    sys.exit(1)

# ── 3. Smoke test ────────────────────────────────────────────
print("\nRunning smoke test (1 sample)...", flush=True)
try:
    _in = tokenizer("Answer yes or no: is 2+2=4?", return_tensors="pt").to(device)
    _out = model.generate(**_in, max_new_tokens=3)
    _dec = tokenizer.decode(_out[0], skip_special_tokens=True)
    print(f"✓ Smoke test passed — model output: '{_dec}'", flush=True)
except Exception as e:
    print(f"✗ Smoke test failed: {e}, falling back to CPU", flush=True)
    device = torch.device("cpu")
    model = model.to(device)

# ── 4. Load data ─────────────────────────────────────────────
DATA_PATH = "data/processed/test_labeled.csv"
if not os.path.exists(DATA_PATH):
    for p in ["test_labeled.csv", "data/test_labeled.csv", "processed/test_labeled.csv"]:
        if os.path.exists(p):
            DATA_PATH = p
            break
    else:
        print("✗ Could not find data file.", flush=True)
        sys.exit(1)

print(f"\nLoading data from: {DATA_PATH}", flush=True)
df = pd.read_csv(DATA_PATH)   # default c-engine, fastest
print(f"  → raw shape: {df.shape}", flush=True)
df = df.dropna(subset=["text", "action_item"])
df["action_item"] = df["action_item"].astype(int)
print(f"✓ Loaded {len(df)} rows | labels: {df['action_item'].value_counts().to_dict()}", flush=True)

# ── 5. Sample ────────────────────────────────────────────────
N = min(200, len(df))
BATCH_SIZE = 8

sample = df.sample(N, random_state=42)
print(f"✓ Sampled {N} | labels: {sample['action_item'].value_counts().to_dict()}", flush=True)

# ── 6. Batched prediction ────────────────────────────────────
def make_prompt(text: str) -> str:
    return (
        "Is the following meeting utterance an action item "
        "(a task assigned to someone)? Answer yes or no only.\n"
        f"Utterance: {text}\nAnswer:"
    )

@torch.no_grad()
def predict_batch(texts: list) -> list:
    prompts = [make_prompt(t) for t in texts]
    inputs = tokenizer(
        prompts, return_tensors="pt", truncation=True,
        max_length=128, padding=True,
    ).to(device)
    outputs = model.generate(**inputs, max_new_tokens=3)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [1 if "yes" in r.lower() else 0 for r in decoded]

# ── 7. Run predictions ───────────────────────────────────────
texts = list(sample["text"])
preds = []
total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

print(f"\nRunning predictions: {N} samples, batch_size={BATCH_SIZE}, {total_batches} batches ...", flush=True)

for batch_num, i in enumerate(range(0, len(texts), BATCH_SIZE), 1):
    batch = texts[i : i + BATCH_SIZE]
    preds.extend(predict_batch(batch))
    done = min(i + BATCH_SIZE, len(texts))
    print(f"  ✓ batch {batch_num}/{total_batches} — {done}/{N} samples", flush=True)

labels = sample["action_item"].tolist()

# ── 8. Results ───────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
print(f"Flan-T5 zero-shot macro-F1: {macro_f1 * 100:.1f}%", flush=True)
print("=" * 60, flush=True)
print(classification_report(
    labels, preds,
    target_names=["Not action", "Action item"],
    zero_division=0,
), flush=True)