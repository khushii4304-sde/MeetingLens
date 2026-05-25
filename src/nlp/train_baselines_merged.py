# train_baselines_merged.py
# =================================================================
# PURPOSE:
#   Retrains TF-IDF + LR and BERT-base on the MERGED training
#   set (AMI + OOD) and evaluates on both AMI and OOD test sets.
#
# WHY THIS IS NEEDED:
#   Without this, the comparison between RoBERTa (trained on
#   merged) and BERT (trained on AMI only) confounds two variables:
#     1. Model architecture (RoBERTa vs BERT)
#     2. Training data (merged vs AMI only)
#
#   By also training BERT on merged data, we can isolate the
#   effect of architecture alone.
#
# SAVES TO:
#   models/tfidf_lr_merged/   ← TF-IDF on merged data
#   models/bert_merged/       ← BERT on merged data
# =================================================================

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
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

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.makedirs("models/tfidf_lr_merged", exist_ok=True)
os.makedirs("models/bert_merged",     exist_ok=True)
os.makedirs("data/processed/results", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65)
print("Training Baselines on Merged AMI + OOD Data")
print("Purpose: isolate model architecture effect from data effect")
print(f"Device: {device}")
print("=" * 65)

# ── Load data ─────────────────────────────────────────────────
print("\nLoading datasets...")
tr   = pd.read_csv(
    "data/processed/train_merged.csv"
).dropna(subset=["text"])

# Cap training size to keep runtime manageable
# 60,000 samples = same as RoBERTa, ~7500 batches/epoch
MAX_TRAIN = 60_000
if len(tr) > MAX_TRAIN:
    # Stratified sample — keep class balance intact
    pos = tr[tr["action_item"] == 1]
    neg = tr[tr["action_item"] == 0]
    pos_ratio = len(pos) / len(tr)
    n_pos = int(MAX_TRAIN * pos_ratio)
    n_neg = MAX_TRAIN - n_pos
    tr = pd.concat([
        pos.sample(n=min(n_pos, len(pos)), random_state=42),
        neg.sample(n=min(n_neg, len(neg)), random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Capped train to {len(tr):,} samples "
          f"(pos: {tr['action_item'].sum():,}, "
          f"neg: {(tr['action_item']==0).sum():,})")
    
val     = pd.read_csv(
    "data/processed/validation_labeled.csv"
).dropna(subset=["text"])
te_ami  = pd.read_csv(
    "data/processed/test_labeled.csv"
).dropna(subset=["text"])
te_ood  = pd.read_csv(
    "data/processed/icsi_test.csv"
).dropna(subset=["text"])

X_tr    = tr["text"].tolist()
y_tr    = tr["action_item"].tolist()
X_val   = val["text"].tolist()
y_val   = val["action_item"].tolist()
X_ami   = te_ami["text"].tolist()
y_ami   = te_ami["action_item"].tolist()
X_ood   = te_ood["text"].tolist()
y_ood   = te_ood["action_item"].tolist()

print(f"  Train (merged) : {len(X_tr):,} | "
      f"AI rate: {sum(y_tr)/len(y_tr)*100:.1f}%")
print(f"  Val (AMI)      : {len(X_val):,}")
print(f"  Test AMI       : {len(X_ami):,}")
print(f"  Test OOD       : {len(X_ood):,}")


def get_metrics(y_true, y_pred):
    return {
        "macro_f1":  round(f1_score(
            y_true, y_pred,
            average="macro", zero_division=0), 4),
        "macro_pre": round(precision_score(
            y_true, y_pred,
            average="macro", zero_division=0), 4),
        "macro_rec": round(recall_score(
            y_true, y_pred,
            average="macro", zero_division=0), 4),
        "binary_f1": round(f1_score(
            y_true, y_pred,
            average="binary", zero_division=0), 4),
    }


all_results = {}

# ═════════════════════════════════════════════════════════════
# MODEL 1 — TF-IDF + LR on MERGED data
# ═════════════════════════════════════════════════════════════
print(f"\n{'═'*65}")
print("MODEL 1: TF-IDF + LR (trained on AMI + OOD merged)")
print(f"{'═'*65}")

tfidf_m = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    strip_accents="unicode"
)
X_tr_tf  = tfidf_m.fit_transform(X_tr)
X_ami_tf = tfidf_m.transform(X_ami)
X_ood_tf = tfidf_m.transform(X_ood)

print("  Fitting Logistic Regression...")
lr_m = LogisticRegression(
    class_weight="balanced",
    C=1.0,
    max_iter=2000,
    random_state=SEED,
    solver="lbfgs",
    n_jobs=-1
)
lr_m.fit(X_tr_tf, y_tr)

ami_preds  = lr_m.predict(X_ami_tf)
ood_preds  = lr_m.predict(X_ood_tf)
lr_ami_met = get_metrics(y_ami, ami_preds)
lr_ood_met = get_metrics(y_ood, ood_preds)
lr_gap     = lr_ami_met["macro_f1"] - lr_ood_met["macro_f1"]

print(f"  AMI  test macro-F1 : {lr_ami_met['macro_f1']*100:.1f}%")
print(f"  OOD  test macro-F1 : {lr_ood_met['macro_f1']*100:.1f}%")
print(f"  Generalisation gap : {lr_gap*100:.1f}%")

print(f"\n  AMI classification report:")
print(classification_report(
    y_ami, ami_preds,
    target_names=["Not Action Item", "Action Item"],
    zero_division=0
))

# Save
with open("models/tfidf_lr_merged/tfidf_vectorizer.pkl","wb") as f:
    pickle.dump(tfidf_m, f)
with open("models/tfidf_lr_merged/lr_model.pkl", "wb") as f:
    pickle.dump(lr_m, f)

results_lr = {
    "model":      "TF-IDF + LR",
    "train_data": "AMI+OOD merged",
    "ami_test":   lr_ami_met,
    "ood_test":   lr_ood_met,
    "gap":        round(lr_gap, 4),
}
with open("models/tfidf_lr_merged/test_results.json", "w") as f:
    json.dump(results_lr, f, indent=2)

all_results["tfidf_lr_merged"] = results_lr
print(f"  Saved to models/tfidf_lr_merged/")

# ═════════════════════════════════════════════════════════════
# MODEL 2 — BERT-base on MERGED data
# ═════════════════════════════════════════════════════════════
print(f"\n{'═'*65}")
print("MODEL 2: BERT-base-uncased (trained on AMI + OOD merged)")
print("This will take 60-90 minutes on your GPU.")
print(f"{'═'*65}")

MAX_LEN  = 96
BATCH    = 8
ACCUM    = 2
EPOCHS   = 3
LR       = 2e-5
PATIENCE = 2

cw = compute_class_weight(
    "balanced",
    classes=np.array([0, 1]),
    y=np.array(y_tr)
)
class_weights = torch.tensor(cw, dtype=torch.float).to(device)
print(f"\n  Class weights: [{cw[0]:.3f}, {cw[1]:.3f}]")


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


MODEL_NAME = "bert-base-uncased"
print(f"\n  Loading {MODEL_NAME}...")
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
).to(device)
print(f"  Parameters: "
      f"{sum(p.numel() for p in model.parameters()):,}")

tr_dl  = DataLoader(
    UtteranceDataset(X_tr,  y_tr,  tokenizer),
    batch_size=BATCH, shuffle=True
)
val_dl = DataLoader(
    UtteranceDataset(X_val, y_val, tokenizer),
    batch_size=BATCH*2, shuffle=False
)
ami_dl = DataLoader(
    UtteranceDataset(X_ami, y_ami, tokenizer),
    batch_size=BATCH*2, shuffle=False
)
ood_dl = DataLoader(
    UtteranceDataset(X_ood, y_ood, tokenizer),
    batch_size=BATCH*2, shuffle=False
)

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
scheduler    = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

print(f"  Total training steps : {total_steps:,}")
print(f"  Starting training ({len(tr_dl):,} batches/epoch)...")

best_val_f1  = 0.0
patience_cnt = 0
best_ami_met = None
best_ood_met = None
best_ami_l   = None
best_ami_p   = None
history      = []

for epoch in range(1, EPOCHS + 1):
    print(f"\n  Epoch {epoch}/{EPOCHS}")
    print("  " + "─" * 43)

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
            print(f"    Step {step+1:>5}/{len(tr_dl)} | "
                  f"Loss: {total_loss/n_batches:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

    avg_loss = total_loss / n_batches

    val_l, val_p = evaluate(model, val_dl)
    val_f1 = f1_score(
        val_l, val_p,
        average="macro", zero_division=0
    )
    print(f"  Avg loss     : {avg_loss:.4f}")
    print(f"  Val macro-F1 : {val_f1:.4f} ({val_f1*100:.1f}%)")

    history.append({
        "epoch":        epoch,
        "train_loss":   round(avg_loss, 4),
        "val_macro_f1": round(val_f1,   4),
    })

    if val_f1 > best_val_f1:
        best_val_f1  = val_f1
        patience_cnt = 0
        model.save_pretrained("models/bert_merged")
        tokenizer.save_pretrained("models/bert_merged")
        print(f"  ✓ Best model saved to models/bert_merged/")

        ami_l, ami_p = evaluate(model, ami_dl)
        ood_l, ood_p = evaluate(model, ood_dl)
        best_ami_met = get_metrics(ami_l, ami_p)
        best_ood_met = get_metrics(ood_l, ood_p)
        best_ami_l, best_ami_p = ami_l, ami_p
    else:
        patience_cnt += 1
        print(f"  Patience: {patience_cnt}/{PATIENCE}")
        if patience_cnt >= PATIENCE:
            print("  Early stopping.")
            break

# Final results
print(f"\n{'='*65}")
print("BERT (MERGED) FINAL TEST RESULTS")
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

results_bert = {
    "model":             "BERT-base-uncased",
    "train_data":        "AMI+OOD merged",
    "best_val_macro_f1": round(best_val_f1, 4),
    "ami_test":          best_ami_met or {},
    "ood_test":          best_ood_met or {},
    "training_history":  history,
}
with open("models/bert_merged/test_results.json", "w") as f:
    json.dump(results_bert, f, indent=2)
with open("models/bert_merged/training_history.json","w") as f:
    json.dump(history, f, indent=2)

all_results["bert_merged"] = results_bert

# ── Final comparison ──────────────────────────────────────────
print(f"\n\n{'='*65}")
print("BASELINE MERGED TRAINING SUMMARY — SCREENSHOT THIS")
print(f"{'='*65}\n")

rows = []
for key, r in all_results.items():
    ami = r.get("ami_test", {})
    ood = r.get("ood_test", {})
    rows.append({
        "Model":       r["model"],
        "Train Data":  r["train_data"],
        "AMI F1 (%)":  round(ami.get("macro_f1",0)*100, 1),
        "OOD F1 (%)":  round(ood.get("macro_f1",0)*100, 1),
        "Gap (%)":     round(r.get("gap",
            ami.get("macro_f1",0) -
            ood.get("macro_f1",0))*100, 1),
    })

import pandas as pd
df = pd.DataFrame(rows)
print(df.to_string(index=False))

df.to_csv(
    "data/processed/results/baselines_merged_results.csv",
    index=False
)
with open(
    "data/processed/results/baselines_merged_results.json",
    "w"
) as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved: data/processed/results/baselines_merged_results.csv")
print("\nNow update generate_final_table.py to include these results.")
print("Phase: Baselines Merged COMPLETE.")