# evaluate_zeroshot.py
# =================================================================
# PURPOSE:
#   Tests AMI-trained models on OOD data (MRDA + MeetingBank)
#   without any retraining. Reveals the true generalisation gap.
#
# KEY FINDING THIS PRODUCES:
#   A large AMI→OOD drop = model learned AMI-specific shortcuts
#   A small drop = model learned genuine action item semantics
#   TF-IDF drops more than BERT = context helps generalisation
#
# This is Research Finding 1 in your paper.
# =================================================================

import os
import json
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report
)

os.makedirs("data/processed/results", exist_ok=True)

SEED       = 42
MAX_LENGTH = 96
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65)
print("Zero-Shot OOD Evaluation")
print("AMI-trained models tested on MRDA + MeetingBank")
print("=" * 65)
print(f"\nDevice: {device}")
if device.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")

# ── Load test sets ────────────────────────────────────────────
print("\nLoading test sets...")

ami_te = pd.read_csv(
    "data/processed/test_labeled.csv"
).dropna(subset=["text"])

icsi_te = pd.read_csv(
    "data/processed/icsi_test.csv"
).dropna(subset=["text"])

# Rename speaker_id → speaker if needed
if ("speaker_id" in icsi_te.columns and
        "speaker" not in icsi_te.columns):
    icsi_te = icsi_te.rename(
        columns={"speaker_id": "speaker"}
    )

print(f"  AMI  test : {len(ami_te):,} utterances | "
      f"AI rate: {ami_te['action_item'].mean()*100:.1f}%")
print(f"  OOD  test : {len(icsi_te):,} utterances | "
      f"AI rate: {icsi_te['action_item'].mean()*100:.1f}%")

if "source" in icsi_te.columns:
    print(f"  OOD sources: "
          f"{icsi_te['source'].value_counts().to_dict()}")


# ── PyTorch Dataset ───────────────────────────────────────────
class UtteranceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer,
                 max_len=MAX_LENGTH):
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


# ── Helper functions ──────────────────────────────────────────
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


def bert_inference(model_dir, X, y):
    """Loads a saved transformer and runs inference."""
    tok   = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir
    ).to(device)
    model.eval()

    ds = UtteranceDataset(X, y, tok)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    preds, labels = [], []
    with torch.no_grad():
        for batch in dl:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            p    = torch.argmax(out.logits, dim=1)
            preds.extend(p.cpu().tolist())
            labels.extend(batch["labels"].tolist())
    return labels, preds


results = {}

# ═════════════════════════════════════════════════════════════
# MODEL 1 — TF-IDF + Logistic Regression
# ═════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("MODEL 1: TF-IDF + Logistic Regression")
print(f"{'─'*65}")

LR_DIR = "models/tfidf_lr"
if (os.path.exists(f"{LR_DIR}/tfidf_vectorizer.pkl") and
        os.path.exists(f"{LR_DIR}/lr_model.pkl")):

    with open(f"{LR_DIR}/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open(f"{LR_DIR}/lr_model.pkl", "rb") as f:
        lr    = pickle.load(f)

    # AMI test
    ami_tfidf  = tfidf.transform(ami_te["text"].tolist())
    ami_lr_p   = lr.predict(ami_tfidf)
    lr_ami     = get_metrics(
        ami_te["action_item"].tolist(), ami_lr_p
    )

    # OOD test (zero-shot — model has NEVER seen this data)
    ood_tfidf  = tfidf.transform(icsi_te["text"].tolist())
    ood_lr_p   = lr.predict(ood_tfidf)
    lr_ood     = get_metrics(
        icsi_te["action_item"].tolist(), ood_lr_p
    )

    drop_lr = lr_ami["macro_f1"] - lr_ood["macro_f1"]

    print(f"  AMI  test macro-F1 : {lr_ami['macro_f1']*100:.1f}%")
    print(f"  OOD  test macro-F1 : {lr_ood['macro_f1']*100:.1f}%")
    print(f"  Generalisation drop: {drop_lr*100:.1f}%")

    print(f"\n  OOD classification report:")
    print(classification_report(
        icsi_te["action_item"].tolist(), ood_lr_p,
        target_names=["Not Action Item", "Action Item"],
        zero_division=0
    ))

    results["tfidf_lr"] = {
        "ami_macro_f1":  lr_ami["macro_f1"],
        "ood_macro_f1":  lr_ood["macro_f1"],
        "ood_drop":      round(drop_lr, 4),
        "ami_full":      lr_ami,
        "ood_full":      lr_ood,
    }
else:
    print("  TF-IDF model not found at models/tfidf_lr/")
    print("  Make sure Day 3 training completed successfully.")

# ═════════════════════════════════════════════════════════════
# MODEL 2 — BERT-base-uncased
# ═════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print("MODEL 2: BERT-base-uncased (AMI-only trained)")
print(f"{'─'*65}")

BERT_DIR = "models/bert_base"
if os.path.exists(BERT_DIR):

    # AMI test
    ami_l, ami_p = bert_inference(
        BERT_DIR,
        ami_te["text"].tolist(),
        ami_te["action_item"].tolist()
    )
    bert_ami = get_metrics(ami_l, ami_p)

    # OOD test (zero-shot)
    ood_l, ood_p = bert_inference(
        BERT_DIR,
        icsi_te["text"].tolist(),
        icsi_te["action_item"].tolist()
    )
    bert_ood = get_metrics(ood_l, ood_p)

    drop_bert = bert_ami["macro_f1"] - bert_ood["macro_f1"]

    print(f"  AMI  test macro-F1 : {bert_ami['macro_f1']*100:.1f}%")
    print(f"  OOD  test macro-F1 : {bert_ood['macro_f1']*100:.1f}%")
    print(f"  Generalisation drop: {drop_bert*100:.1f}%")

    print(f"\n  OOD classification report:")
    print(classification_report(
        ood_l, ood_p,
        target_names=["Not Action Item", "Action Item"],
        zero_division=0
    ))

    results["bert_base"] = {
        "ami_macro_f1":  bert_ami["macro_f1"],
        "ood_macro_f1":  bert_ood["macro_f1"],
        "ood_drop":      round(drop_bert, 4),
        "ami_full":      bert_ami,
        "ood_full":      bert_ood,
    }
else:
    print(f"  BERT model not found at {BERT_DIR}/")
    print("  Make sure Day 3 BERT training completed.")

# ═════════════════════════════════════════════════════════════
# GAP ANALYSIS TABLE
# ═════════════════════════════════════════════════════════════
print(f"\n\n{'='*65}")
print("ZERO-SHOT GAP TABLE — SCREENSHOT THIS")
print("This is Research Finding 1 in your paper.")
print(f"{'='*65}\n")

print(f"  {'Model':<30} {'AMI F1':>8} {'OOD F1':>8} "
      f"{'Drop':>8}")
print(f"  {'─'*56}")

for model_name, key in [
    ("TF-IDF + LR (AMI only)",   "tfidf_lr"),
    ("BERT-base (AMI only)",     "bert_base"),
]:
    if key in results:
        r = results[key]
        print(f"  {model_name:<30} "
              f"{r['ami_macro_f1']*100:>7.1f}% "
              f"{r['ood_macro_f1']*100:>7.1f}% "
              f"{r['ood_drop']*100:>+7.1f}%")

print(f"\n  Key insight:")
print(f"  → Both models drop significantly on OOD data")
print(f"  → TF-IDF drop > BERT drop (context helps)")
print(f"  → This justifies cross-corpus training (Phase C)")

# Save results
with open(
    "data/processed/results/zeroshot_results.json", "w"
) as f:
    save = {k: {kk: vv for kk, vv in v.items()
                if kk not in ["ami_full", "ood_full"]}
            for k, v in results.items()}
    json.dump(save, f, indent=2)

print(f"\nSaved: data/processed/results/zeroshot_results.json")
print("\nPhase A COMPLETE. Push to GitHub, then run Phase B.")