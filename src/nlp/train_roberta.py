# train_roberta.py
# =================================================================
# PURPOSE:
#   Fine-tunes RoBERTa-base on merged AMI+OOD training data.
#   Evaluates on both AMI test set and OOD test set independently.
#
# WHY RoBERTa IS THE PROPOSED MODEL:
#   1. Trained on 160GB text (10x more than BERT's 16GB)
#   2. Removes Next Sentence Prediction (useless for utterances)
#   3. Dynamic masking — sees different word masks each epoch
#      making representations more robust
#   4. Consistently 1-5% F1 better than BERT on classification
#
# CROSS-CORPUS TRAINING HYPOTHESIS:
#   RoBERTa trained on AMI+OOD will show a SMALLER generalisation
#   gap than BERT trained on AMI only.
#   This is Research Finding 2 in your paper.
# =================================================================

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from torch.optim import AdamW
# ── Reproducibility ───────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Configuration ─────────────────────────────────────────────
MODEL_NAME = "roberta-base"
SAVE_DIR   = "models/roberta_base"
MAX_LEN    = 96
BATCH      = 8
ACCUM      = 2       # effective batch = 8 × 2 = 16
EPOCHS     = 2
LR         = 2e-5
PATIENCE   = 2

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65)
print("Training: RoBERTa-base (Proposed Model)")
print(f"Training data  : AMI + OOD (merged)")
print(f"Device         : {device}")
if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU            : {props.name}")
    print(f"GPU Memory     : {props.total_memory/1e9:.1f} GB")
print(f"Max length     : {MAX_LEN} tokens")
print(f"Batch size     : {BATCH} (effective: {BATCH*ACCUM})")
print(f"Max epochs     : {EPOCHS}")
print(f"Learning rate  : {LR}")
print("=" * 65)

# ── Load data ─────────────────────────────────────────────────
print("\nLoading datasets...")

tr   = pd.read_csv(
    "data/processed/train_merged.csv"
).dropna(subset=["text"])

tr = tr.sample(
    n=60000,
    random_state=42
).reset_index(drop=True)

val  = pd.read_csv(
    "data/processed/validation_labeled.csv"
).dropna(subset=["text"])
te_ami = pd.read_csv(
    "data/processed/test_labeled.csv"
).dropna(subset=["text"])
te_ood = pd.read_csv(
    "data/processed/icsi_test.csv"
).dropna(subset=["text"])

print(f"  Train (merged) : {len(tr):,} utterances | "
      f"AI rate: {tr['action_item'].mean()*100:.1f}%")
print(f"  Val (AMI)      : {len(val):,} utterances")
print(f"  Test AMI       : {len(te_ami):,} utterances")
print(f"  Test OOD       : {len(te_ood):,} utterances")

# ── Class weights ─────────────────────────────────────────────
# Computed from merged training set only
# Handles the natural class imbalance
cw = compute_class_weight(
    "balanced",
    classes=np.array([0, 1]),
    y=np.array(tr["action_item"].tolist())
)
class_weights = torch.tensor(cw, dtype=torch.float).to(device)
print(f"\nClass weights from merged training set:")
print(f"  Class 0 (not action): {cw[0]:.4f}")
print(f"  Class 1 (action item): {cw[1]:.4f}")


# ── Dataset class ─────────────────────────────────────────────
class UtteranceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer,
                 max_len=MAX_LEN):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(
                int(self.labels[idx]), dtype=torch.long
            )
        }


# ── Evaluation function ───────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            p    = torch.argmax(out.logits, dim=1)
            preds.extend(p.cpu().tolist())
            labels.extend(batch["labels"].tolist())
    return labels, preds


def get_metrics(labels, preds):
    return {
        "macro_f1":  round(f1_score(
            labels, preds,
            average="macro", zero_division=0), 4),
        "macro_pre": round(precision_score(
            labels, preds,
            average="macro", zero_division=0), 4),
        "macro_rec": round(recall_score(
            labels, preds,
            average="macro", zero_division=0), 4),
        "binary_f1": round(f1_score(
            labels, preds,
            average="binary", zero_division=0), 4),
    }


# ── DataLoaders ───────────────────────────────────────────────
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tr_dl  = DataLoader(
    UtteranceDataset(
        tr["text"].tolist(),
        tr["action_item"].tolist(), tokenizer
    ),
    batch_size=BATCH, shuffle=True
)
val_dl = DataLoader(
    UtteranceDataset(
        val["text"].tolist(),
        val["action_item"].tolist(), tokenizer
    ),
    batch_size=BATCH*2, shuffle=False
)
ami_dl = DataLoader(
    UtteranceDataset(
        te_ami["text"].tolist(),
        te_ami["action_item"].tolist(), tokenizer
    ),
    batch_size=BATCH*2, shuffle=False
)
ood_dl = DataLoader(
    UtteranceDataset(
        te_ood["text"].tolist(),
        te_ood["action_item"].tolist(), tokenizer
    ),
    batch_size=BATCH*2, shuffle=False
)

print(f"Training batches   : {len(tr_dl):,}")
print(f"Validation batches : {len(val_dl):,}")

# ── Load model ────────────────────────────────────────────────
print(f"\nLoading {MODEL_NAME} from HuggingFace...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True
).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

# ── Optimiser ─────────────────────────────────────────────────
# Weight decay on all params except biases and LayerNorm
no_decay   = ["bias", "LayerNorm.weight"]
opt_params = [
    {
        "params": [p for n, p in model.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01
    },
    {
        "params": [p for n, p in model.named_parameters()
                   if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0
    },
]
optimizer    = AdamW(opt_params, lr=LR)
total_steps  = (len(tr_dl) // ACCUM) * EPOCHS
warmup_steps = int(total_steps * 0.1)
scheduler    = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

print(f"Total training steps : {total_steps:,}")
print(f"Warmup steps         : {warmup_steps:,}")

# ── Training loop ─────────────────────────────────────────────
best_val_f1  = 0.0
patience_cnt = 0
best_ami_met = None
best_ood_met = None
best_ami_l   = None
best_ami_p   = None
history      = []

print(f"\n{'─'*65}")
print("Starting training...")
print(f"{'─'*65}")

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    print("─" * 45)

    # Training pass
    model.train()
    total_loss = 0.0
    n_batches  = 0
    optimizer.zero_grad()

    for step, batch in enumerate(tr_dl):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labs = batch["labels"].to(device)

        out  = model(input_ids=ids, attention_mask=mask)
        loss = loss_fn(out.logits, labs) / ACCUM
        loss.backward()

        total_loss += loss.item() * ACCUM
        n_batches  += 1

        if (step + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 200 == 0:
            avg  = total_loss / n_batches
            lr_n = scheduler.get_last_lr()[0]
            print(f"  Step {step+1:>5}/{len(tr_dl)} | "
                  f"Loss: {avg:.4f} | LR: {lr_n:.2e}")

    avg_loss = total_loss / n_batches

    # Validation pass
    val_l, val_p = evaluate(model, val_dl)
    val_f1 = f1_score(
        val_l, val_p,
        average="macro", zero_division=0
    )
    print(f"  Avg train loss : {avg_loss:.4f}")
    print(f"  Val macro-F1   : {val_f1:.4f} "
          f"({val_f1*100:.1f}%)")

    history.append({
        "epoch":        epoch,
        "train_loss":   round(avg_loss, 4),
        "val_macro_f1": round(val_f1,   4),
    })

    # Early stopping
    if val_f1 > best_val_f1:
        best_val_f1  = val_f1
        patience_cnt = 0

        # Save best checkpoint
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"  ✓ New best! Model saved to {SAVE_DIR}/")

        # Evaluate on both test sets with best model
        ami_l, ami_p = evaluate(model, ami_dl)
        ood_l, ood_p = evaluate(model, ood_dl)

        best_ami_met = get_metrics(ami_l, ami_p)
        best_ood_met = get_metrics(ood_l, ood_p)
        best_ami_l, best_ami_p = ami_l, ami_p
    else:
        patience_cnt += 1
        print(f"  No improvement. "
              f"Patience: {patience_cnt}/{PATIENCE}")
        if patience_cnt >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}.")
            break

# ── Final results ─────────────────────────────────────────────
print(f"\n{'='*65}")
print("ROBERTA FINAL TEST RESULTS")
print(f"{'='*65}")

if best_ami_met:
    gap = (best_ami_met["macro_f1"] -
           best_ood_met["macro_f1"])

    print(f"\n  AMI  test macro-F1 : "
          f"{best_ami_met['macro_f1']*100:.1f}%")
    print(f"  OOD  test macro-F1 : "
          f"{best_ood_met['macro_f1']*100:.1f}%")
    print(f"  Generalisation gap : {gap*100:.1f}%")

    print(f"\n  AMI detailed report:")
    print(classification_report(
        best_ami_l, best_ami_p,
        target_names=["Not Action Item", "Action Item"],
        zero_division=0
    ))

# ── Save results ──────────────────────────────────────────────
results = {
    "model":             "RoBERTa-base",
    "train_data":        "AMI+OOD (merged)",
    "best_val_macro_f1": round(best_val_f1, 4),
    "ami_test":          best_ami_met or {},
    "ood_test":          best_ood_met or {},
    "training_history":  history,
    "config": {
        "max_length":        MAX_LEN,
        "batch_size":        BATCH,
        "accumulation":      ACCUM,
        "effective_batch":   BATCH * ACCUM,
        "learning_rate":     LR,
        "max_epochs":        EPOCHS,
        "patience":          PATIENCE,
        "seed":              SEED,
    }
}
with open(f"{SAVE_DIR}/test_results.json", "w") as f:
    json.dump(results, f, indent=2)
with open(f"{SAVE_DIR}/training_history.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\nAll results saved to {SAVE_DIR}/")
print("Phase C COMPLETE. Push to GitHub, then run Phase D.")