# src/nlp/gpt_baseline.py

import sys
import os
import io
import json
import warnings
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("=" * 60, flush=True)
print("Flan-T5 Zero-Shot Baseline", flush=True)
print("=" * 60, flush=True)

# ── 1. Load data FIRST using chunked read ────────────────────
print("Loading data...", flush=True)
chunks = []
for chunk in pd.read_csv(
    "data/processed/test_labeled.csv",
    chunksize=1000,
    encoding='utf-8',
    encoding_errors='replace'
):
    chunks.append(chunk)
df = pd.concat(chunks).dropna(subset=["text"])
print(f"Rows loaded: {len(df)}", flush=True)
print(f"Label distribution: {df['action_item'].value_counts().to_dict()}", flush=True)

N = 200
sample = df.sample(N, random_state=42).reset_index(drop=True)
texts  = sample["text"].tolist()
labels = sample["action_item"].tolist()
print(f"Sampled {N} examples. Positives: {sum(labels)}", flush=True)

# ── 2. Load model AFTER data ──────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}", flush=True)

MODEL_NAME = "google/flan-t5-base"
print(f"Loading {MODEL_NAME}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval()
torch.cuda.empty_cache()
print("Model loaded.", flush=True)

# ── 3. Predict ────────────────────────────────────────────────
@torch.no_grad()
def predict(text):
    prompt = (
        "Task: Classify whether this meeting utterance is an action item.\n"
        "An action item is a specific task assigned to someone to complete.\n"
        f"Utterance: {text}\n"
        "Is this an action item? Answer with yes or no."
    )
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=256
    ).to(device)
    outputs = model.generate(**inputs, max_new_tokens=3)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    return 1 if answer.startswith("yes") else 0

print("\nRunning predictions...", flush=True)
preds = []
for i, text in enumerate(texts):
    preds.append(predict(text))
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N} done...", flush=True)

print(f"\nPrediction distribution: {preds.count(0)} zeros, {preds.count(1)} ones", flush=True)

# ── 4. Results ────────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("FLAN-T5 ZERO-SHOT RESULTS", flush=True)
print("=" * 60, flush=True)

macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
print(f"Macro-F1 : {macro_f1 * 100:.1f}%", flush=True)
print(f"Samples  : {N}", flush=True)
print("\nClassification report:", flush=True)
print(classification_report(
    labels, preds,
    target_names=["Not action item", "Action item"],
    zero_division=0
), flush=True)

# ── 5. Save ───────────────────────────────────────────────────
os.makedirs("data/processed/results", exist_ok=True)
result = {
    "model":      "Flan-T5-base (zero-shot)",
    "samples":    N,
    "macro_f1":   round(macro_f1, 4),
    "pred_ones":  preds.count(1),
    "pred_zeros": preds.count(0),
}
with open("data/processed/results/flant5_results.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\nSaved: data/processed/results/flant5_results.json", flush=True)
print("=" * 60, flush=True)