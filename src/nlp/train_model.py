# train_model.py
# =============================================================================
# PURPOSE:
#   Trains and compares 4 models for action item extraction from meeting
#   transcripts. Uses the full AMI corpus (~150k utterances).
#
# MODELS:
#   1. TF-IDF + Logistic Regression  — classical ML baseline
#   2. BERT-base-uncased             — strong transformer baseline
#   3. RoBERTa-base                  — PROPOSED model (best expected)
#   4. DistilBERT-base-uncased       — efficient transformer variant
#
# PRIMARY METRIC: Macro-F1 Score
#   We use MACRO-F1 (not binary F1) because it averages F1 across both
#   classes equally, penalising models that ignore the minority class.
#   This is the standard metric for imbalanced classification in NLP papers.
#
# TRAINING DECISIONS:
#   - Class weights: handles 15-25% positive class imbalance
#   - AdamW optimizer: standard for transformer fine-tuning
#   - Linear warmup schedule: prevents unstable early training
#   - Gradient accumulation: simulates larger batches on 8GB GPU
#   - Early stopping on validation macro-F1: prevents overfitting
#   - random seed 42: ensures reproducibility
# =============================================================================

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    f1_score, precision_score, recall_score, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from torch.optim import AdamW

# =============================================================================
# REPRODUCIBILITY
# Setting all random seeds ensures your results are identical every run.
# This is required for research — reviewers must reproduce your numbers.
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =============================================================================
# CONFIGURATION
# =============================================================================
MAX_LENGTH          = 96   # max tokens per utterance
                            # AMI utterances avg ~20 words, 96 is safe
BATCH_SIZE          = 8    # samples per GPU batch
                            # 8 is safe for 8GB VRAM with BERT-class models
ACCUMULATION_STEPS  = 2     # accumulate 2 batches before weight update
                            
NUM_EPOCHS          = 4     # max epochs (early stopping may stop sooner)
LEARNING_RATE       = 1e-5  # standard for transformer fine-tuning
                            # too high → unstable; too low → slow convergence
WARMUP_RATIO        = 0.1   # first 10% of steps used for LR warmup
PATIENCE            = 2     # early stopping: stop if val F1 doesn't improve
                            # for 2 consecutive epochs
WEIGHT_DECAY        = 0.05  # L2 regularisation to prevent overfitting

os.makedirs("models",                    exist_ok=True)
os.makedirs("data/processed/results",   exist_ok=True)

# =============================================================================
# DEVICE SETUP
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65)
print("MeetingLens — Day 3: Research-Level Model Training")
print("=" * 65)
print(f"\nDevice        : {device}")
if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU name      : {props.name}")
    print(f"GPU memory    : {props.total_memory / 1e9:.1f} GB")
    print(f"CUDA version  : {torch.version.cuda}")
print(f"\nConfiguration :")
print(f"  Max length          : {MAX_LENGTH} tokens")
print(f"  Batch size          : {BATCH_SIZE}")
print(f"  Accumulation steps  : {ACCUMULATION_STEPS}")
print(f"  Effective batch     : {BATCH_SIZE * ACCUMULATION_STEPS}")
print(f"  Max epochs          : {NUM_EPOCHS}")
print(f"  Learning rate       : {LEARNING_RATE}")
print(f"  Early stop patience : {PATIENCE}")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\nLoading labeled data...")

tr  = pd.read_csv("data/processed/train_labeled_hardneg.csv").dropna(subset=["text"])
val = pd.read_csv("data/processed/validation_labeled.csv").dropna(subset=["text"])
te  = pd.read_csv("data/processed/test_labeled.csv").dropna(subset=["text"])

# =============================================================================
# CHECK FOR MEETING LEAKAGE
# =============================================================================

train_meetings = set(tr["meeting_id"].unique())
val_meetings   = set(val["meeting_id"].unique())
test_meetings  = set(te["meeting_id"].unique())

print("\nChecking meeting overlap between splits...")

print("Train-Val overlap :", len(train_meetings & val_meetings))
print("Train-Test overlap:", len(train_meetings & test_meetings))
print("Val-Test overlap  :", len(val_meetings & test_meetings))

X_tr,  y_tr  = tr["text"].tolist(),  tr["action_item"].tolist()
X_val, y_val = val["text"].tolist(), val["action_item"].tolist()
X_te,  y_te  = te["text"].tolist(),  te["action_item"].tolist()

print(f"\nDataset sizes:")
print(f"  Train      : {len(X_tr):,} utterances")
print(f"  Validation : {len(X_val):,} utterances")
print(f"  Test       : {len(X_te):,} utterances")
print(f"  Action item rate (train) : {sum(y_tr)/len(y_tr)*100:.1f}%")
print(f"  Action item rate (test)  : {sum(y_te)/len(y_te)*100:.1f}%")

# =============================================================================
# CLASS WEIGHTS
# Computed from training data only — never from val or test.
# Formula: weight_class_i = n_total / (n_classes × n_class_i)
# Effect:  rare class (action items) gets higher weight → model penalised
#          more for missing action items than for false positives
# =============================================================================
cw = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=np.array(y_tr)
)
class_weights = torch.tensor(cw, dtype=torch.float).to(device)
print(f"\nClass weights (for imbalance handling):")
print(f"  Class 0 (not action item) : {cw[0]:.4f}")
print(f"  Class 1 (action item)     : {cw[1]:.4f}")

# =============================================================================
# PYTORCH DATASET
# =============================================================================
class UtteranceDataset(Dataset):
    """
    PyTorch Dataset wrapper for utterance text + labels.

    __len__    : returns total number of samples
    __getitem__: returns one tokenized sample at index idx

    Tokenization converts text to numbers:
        "Can you send the report?" →
        input_ids      = [101, 2064, 2017, 4604, 1996, 3189, 1029, 102]
        attention_mask = [1,   1,    1,    1,    1,    1,    1,    1  ]

    input_ids      : token IDs from the model vocabulary
    attention_mask : 1 = real token, 0 = padding (we ignore padding)
    padding        : all sequences padded to MAX_LENGTH for batch processing
    truncation     : sequences longer than MAX_LENGTH are cut
    """
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LENGTH):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(int(self.labels[idx]),
                                           dtype=torch.long)
        }

# =============================================================================
# EVALUATION FUNCTION
# =============================================================================
def evaluate_model(model, data_loader, device):
    """
    Runs the model on a dataset and returns predictions and true labels.
    Uses torch.no_grad() to disable gradient computation during evaluation
    — this saves memory and speeds up evaluation significantly.
    model.eval() disables dropout layers that are only active during training.
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)

            outputs   = model(
                input_ids=input_ids,
                attention_mask=attn_mask
            )
            # logits: raw scores before softmax
            # argmax: pick the class with highest score
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return all_preds, all_labels


def compute_metrics(labels, preds, model_name, split_name):
    """
    Computes and prints all evaluation metrics.
    PRIMARY METRIC: macro-F1
    SECONDARY: precision, recall (both macro-averaged)

    Macro-averaging: compute metric for each class separately,
    then take the unweighted average.
    This treats both classes equally regardless of size.
    """
    macro_f1  = f1_score(labels, preds, average="macro",  zero_division=0)
    macro_pre = precision_score(labels, preds, average="macro",  zero_division=0)
    macro_rec = recall_score(labels, preds, average="macro",  zero_division=0)
    bin_f1    = f1_score(labels, preds, average="binary", zero_division=0)
    bin_pre   = precision_score(labels, preds, average="binary", zero_division=0)
    bin_rec   = recall_score(labels, preds, average="binary", zero_division=0)

    print(f"\n  [{model_name}] {split_name} results:")
    print(f"    Macro-F1 (PRIMARY) : {macro_f1:.4f}  ({macro_f1*100:.1f}%)")
    print(f"    Macro-Precision    : {macro_pre:.4f}  ({macro_pre*100:.1f}%)")
    print(f"    Macro-Recall       : {macro_rec:.4f}  ({macro_rec*100:.1f}%)")
    print(f"    Binary-F1          : {bin_f1:.4f}  ({bin_f1*100:.1f}%)")
    print(f"    Binary-Precision   : {bin_pre:.4f}  ({bin_pre*100:.1f}%)")
    print(f"    Binary-Recall      : {bin_rec:.4f}  ({bin_rec*100:.1f}%)")

    return {
        "macro_f1":        round(macro_f1,  4),
        "macro_precision": round(macro_pre, 4),
        "macro_recall":    round(macro_rec, 4),
        "binary_f1":       round(bin_f1,    4),
        "binary_precision":round(bin_pre,   4),
        "binary_recall":   round(bin_rec,   4),
    }

# =============================================================================
# TRANSFORMER TRAINING FUNCTION
# Works for ANY HuggingFace model — BERT, RoBERTa, DistilBERT
# =============================================================================
def train_transformer(
    model_name,
    display_name,
    save_dir,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE
):
    """
    Fine-tunes a HuggingFace transformer for action item classification.

    TRAINING LOOP:
    For each epoch:
        For each batch:
            1. Forward pass  : model processes input, produces logits
            2. Loss          : cross-entropy weighted by class_weights
            3. Backward pass : compute gradients via backpropagation
            4. Accumulate    : if not accumulation boundary, skip update
            5. Clip gradients: prevent exploding gradients (max_norm=1.0)
            6. Update weights: AdamW step
            7. Update LR     : scheduler step

    After each epoch:
        Evaluate on validation set
        If val macro-F1 improved: save model checkpoint
        If no improvement for PATIENCE epochs: stop training

    Parameters:
        model_name   : HuggingFace model identifier (e.g. "roberta-base")
        display_name : Human-readable name for printing
        save_dir     : Where to save the trained model
        epochs       : Maximum training epochs
        batch_size   : Samples per batch
        lr           : Learning rate
    """
    print(f"\n{'═'*65}")
    print(f"Training: {display_name}")
    print(f"Model   : {model_name}")
    print(f"{'═'*65}")

    # Load tokenizer and model
    print(f"  Loading tokenizer and model from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,           # binary: action item or not
        ignore_mismatched_sizes=True
    )
    # Increase dropout to reduce overfitting
    if hasattr(model.config, "hidden_dropout_prob"):
        model.config.hidden_dropout_prob = 0.3

    if hasattr(model.config, "attention_probs_dropout_prob"):
        model.config.attention_probs_dropout_prob = 0.3

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Create datasets and dataloaders
    tr_dataset  = UtteranceDataset(X_tr,  y_tr,  tokenizer)
    val_dataset = UtteranceDataset(X_val, y_val, tokenizer)
    te_dataset  = UtteranceDataset(X_te,  y_te,  tokenizer)

    tr_loader   = DataLoader(
        tr_dataset,  batch_size=batch_size,
        shuffle=True,  num_workers=0, pin_memory=True
    )
    val_loader  = DataLoader(
        val_dataset, batch_size=batch_size*2,
        shuffle=False, num_workers=0, pin_memory=True
    )
    te_loader   = DataLoader(
        te_dataset,  batch_size=batch_size*2,
        shuffle=False, num_workers=0, pin_memory=True
    )

    print(f"  Training batches   : {len(tr_loader):,}")
    print(f"  Validation batches : {len(val_loader):,}")

    # Optimizer
    # Weight decay applied to all parameters except biases and LayerNorm
    # (standard practice — biases/LayerNorm don't benefit from regularisation)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_params = [
        {
            "params":       [p for n, p in model.named_parameters()
                             if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY
        },
        {
            "params":       [p for n, p in model.named_parameters()
                             if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(optimizer_params, lr=lr)

    # Learning rate scheduler with linear warmup
    total_steps  = (len(tr_loader) // ACCUMULATION_STEPS) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    print(f"  Total training steps : {total_steps:,}")
    print(f"  Warmup steps         : {warmup_steps:,}")

    # Weighted cross-entropy loss
    # class_weights computed earlier from training distribution
    loss_fn = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.15
)

    # Early stopping state
    best_val_f1   = 0.0
    patience_cnt  = 0
    best_preds_te = None
    history       = []

    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"\n  Epoch {epoch}/{epochs}")
        print(f"  {'─'*45}")

        # ── Training pass ──
        model.train()
        total_loss   = 0.0
        n_batches    = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tr_loader):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask
            )
            # Weighted loss — penalises missing action items more
            loss = loss_fn(outputs.logits, labels)

            # Scale loss by accumulation steps
            # so the effective gradient magnitude stays consistent
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            total_loss += loss.item() * ACCUMULATION_STEPS
            n_batches  += 1

            # Gradient accumulation — only update every N steps
            if (step + 1) % ACCUMULATION_STEPS == 0:
                # Gradient clipping prevents exploding gradients
                # max_norm=1.0 is standard for transformer fine-tuning
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Print progress every 200 steps
            if (step + 1) % 200 == 0:
                avg_loss = total_loss / n_batches
                lr_now   = scheduler.get_last_lr()[0]
                print(f"    Step {step+1:>5}/{len(tr_loader)} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr_now:.2e}")

        avg_train_loss = total_loss / n_batches
        print(f"  Average training loss : {avg_train_loss:.4f}")

        # ── Validation pass ──
        val_preds, val_labels = evaluate_model(model, val_loader, device)
        val_f1 = f1_score(
            val_labels, val_preds, average="macro", zero_division=0
        )
        val_pre = precision_score(
            val_labels, val_preds, average="macro", zero_division=0
        )
        val_rec = recall_score(
            val_labels, val_preds, average="macro", zero_division=0
        )
        print(f"  Validation macro-F1   : {val_f1:.4f} ({val_f1*100:.1f}%)")
        print(f"  Validation macro-P    : {val_pre:.4f}")
        print(f"  Validation macro-R    : {val_rec:.4f}")

        history.append({
            "epoch":         epoch,
            "train_loss":    round(avg_train_loss, 4),
            "val_macro_f1":  round(val_f1,  4),
            "val_macro_pre": round(val_pre, 4),
            "val_macro_rec": round(val_rec, 4),
        })

        # ── Early stopping ──
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_cnt = 0

            # Save best model checkpoint
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  ✓ New best val F1={val_f1:.4f} — model saved to {save_dir}/")

            # Run test set with best model
            te_preds, te_labels = evaluate_model(model, te_loader, device)
            best_preds_te = (te_preds, te_labels)
        else:
            patience_cnt += 1
            print(f"  No improvement. Patience: {patience_cnt}/{PATIENCE}")
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch}.")
                break

    # ── Final test evaluation ──
    print(f"\n  {'─'*45}")
    print(f"  FINAL TEST RESULTS — {display_name}")
    if best_preds_te is not None:
        te_preds, te_labels = best_preds_te
    else:
        te_preds, te_labels = evaluate_model(model, te_loader, device)

    metrics = compute_metrics(te_labels, te_preds, display_name, "TEST")

    # Detailed classification report
    print(f"\n  Detailed classification report:")
    print(classification_report(
        te_labels, te_preds,
        target_names=["Not Action Item", "Action Item"],
        zero_division=0
    ))

    # Save training history
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save test results
    results = {
        "model":            display_name,
        "hf_name":          model_name,
        "best_val_macro_f1":round(best_val_f1, 4),
        "test_metrics":     metrics,
        "training_history": history,
        "config": {
            "max_length":         MAX_LENGTH,
            "batch_size":         BATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "effective_batch":    BATCH_SIZE * ACCUMULATION_STEPS,
            "learning_rate":      LEARNING_RATE,
            "warmup_ratio":       WARMUP_RATIO,
            "max_epochs":         NUM_EPOCHS,
            "patience":           PATIENCE,
            "seed":               SEED,
        }
    }
    with open(os.path.join(save_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Best val macro-F1 : {best_val_f1:.4f}")
    print(f"  Test macro-F1     : {metrics['macro_f1']:.4f}")
    print(f"  Saved to          : {save_dir}/")

    return metrics, results

# =============================================================================
# ALL RESULTS STORAGE
# =============================================================================
all_results = []

# =============================================================================
# MODEL 1 — TF-IDF + LOGISTIC REGRESSION (classical ML baseline)
# =============================================================================
print(f"\n{'═'*65}")
print("MODEL 1: TF-IDF + Logistic Regression (Classical Baseline)")
print(f"{'═'*65}")
print()
print("  Why this baseline?")
print("  TF-IDF represents text as sparse word-frequency vectors.")
print("  LR draws a linear decision boundary in this feature space.")
print("  No contextual understanding — 'will' always means the same thing.")
print("  Expected F1: 55-65%")
print()

# TF-IDF vectorizer
# max_features=50000 : keep only top 50k most informative features
# ngram_range=(1,2)  : use both single words AND pairs (bigrams)
#                      "will send" is more informative than "will" alone
# sublinear_tf=True  : log(tf) instead of raw tf — reduces impact of
#                      very frequent words
# min_df=2           : ignore words appearing in fewer than 2 documents
print("  Fitting TF-IDF vectorizer (max 50k features, unigrams+bigrams)...")
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    strip_accents="unicode",
    analyzer="word"
)
X_tr_tfidf  = tfidf.fit_transform(X_tr)   # fit on train only
X_val_tfidf = tfidf.transform(X_val)       # transform val
X_te_tfidf  = tfidf.transform(X_te)        # transform test

print(f"  Feature matrix shape: {X_tr_tfidf.shape}")

# Logistic Regression
# class_weight="balanced": equivalent to our class_weights above
# C=1.0: inverse regularisation strength (higher = less regularisation)
# max_iter=2000: enough iterations to converge on large datasets
print("  Training Logistic Regression...")
lr_model = LogisticRegression(
    class_weight="balanced",
    C=1.0,
    max_iter=2000,
    random_state=SEED,
    solver="lbfgs",
    n_jobs=-1
)
lr_model.fit(X_tr_tfidf, y_tr)

# Evaluate on validation
lr_val_preds = lr_model.predict(X_val_tfidf)
lr_val_f1    = f1_score(y_val, lr_val_preds, average="macro", zero_division=0)
print(f"  Validation macro-F1 : {lr_val_f1:.4f} ({lr_val_f1*100:.1f}%)")

# Evaluate on test
lr_te_preds = lr_model.predict(X_te_tfidf)
lr_metrics  = compute_metrics(y_te, lr_te_preds,
                               "TF-IDF + LR", "TEST")

print(f"\n  Detailed classification report:")
print(classification_report(
    y_te, lr_te_preds,
    target_names=["Not Action Item", "Action Item"],
    zero_division=0
))

# ── STEP 1: Feature importance diagnostic ──────────────────────
print("\n" + "="*65)
print("DIAGNOSTIC: What is TF-IDF actually learning?")
print("="*65)

feature_names = tfidf.get_feature_names_out()
coefs = lr_model.coef_[0]

# Top 30 words that predict ACTION ITEM
top_positive = sorted(zip(coefs, feature_names), reverse=True)[:30]
print("\nTop 30 words predicting ACTION ITEM (label=1):")
for score, word in top_positive:
    print(f"  {word:<30} {score:+.3f}")

# Top 30 words that predict NOT action item
top_negative = sorted(zip(coefs, feature_names))[:30]
print("\nTop 30 words predicting NOT action item (label=0):")
for score, word in top_negative:
    print(f"  {word:<30} {score:+.3f}")

# Key question: are modal verbs dominating?
modal_keywords = [
    "will", "should", "need", "going", "action",
    "would", "could", "shall", "must", "can", "gonna"
]


all_results.append({
    "model":            "TF-IDF + Logistic Regression",
    "type":             "Classical Baseline",
    "test_metrics":     lr_metrics,
    "best_val_macro_f1":round(lr_val_f1, 4),
})

# Save LR model info
lr_save = "models/tfidf_lr"
os.makedirs(lr_save, exist_ok=True)
import pickle
with open(os.path.join(lr_save, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)
with open(os.path.join(lr_save, "lr_model.pkl"), "wb") as f:
    pickle.dump(lr_model, f)
with open(os.path.join(lr_save, "test_results.json"), "w") as f:
    json.dump({"model": "TF-IDF + LR", "test_metrics": lr_metrics}, f, indent=2)
print(f"\n  Saved to models/tfidf_lr/")

# =============================================================================
# MODEL 2 — BERT-base-uncased (strong transformer baseline)
# =============================================================================
print(f"\n{'═'*65}")
print("MODEL 2: BERT-base-uncased (Strong Transformer Baseline)")
print(f"{'═'*65}")
print()
print("  Why this baseline?")
print("  BERT introduced bidirectional transformer pre-training in 2019.")
print("  110M parameters. Trained on 16GB of text.")
print("  Bidirectional: reads entire sentence before classifying each word.")
print("  'Can you BANK on that?' vs 'Can you reach the river BANK?'")
print("  BERT understands the difference. TF-IDF does not.")
print("  Expected F1: 72-80%")
print()
print("  NOTE: This will take ~60-90 minutes on your GPU.")
print("  You will see progress every 200 steps.")
print()

bert_metrics, bert_results = train_transformer(
    model_name   = "bert-base-uncased",
    display_name = "BERT-base-uncased",
    save_dir     = "models/bert_base"
)
all_results.append({
    "model":            "BERT-base-uncased",
    "type":             "Strong Baseline",
    "test_metrics":     bert_metrics,
    "best_val_macro_f1":bert_results["best_val_macro_f1"],
})

# =============================================================================
# MODEL 3 — RoBERTa-base (PROPOSED MODEL)
# =============================================================================
# print(f"\n{'═'*65}")
# print("MODEL 3: RoBERTa-base (PROPOSED MODEL ★)")
# print(f"{'═'*65}")
# print()
# print("  Why RoBERTa?")
# print("  Robustly Optimized BERT Pretraining (Liu et al., 2019)")
# print("  Key improvements over BERT:")
# print("    1. Trained on 160GB text (10× more than BERT)")
# print("    2. Removes Next Sentence Prediction (useless for utterances)")
# print("    3. Dynamic masking — different masks each epoch")
# print("    4. Larger batch sizes with longer training")
# print("  Result: 1-5% F1 improvement over BERT on most classification tasks")
# print("  125M parameters.")
# print("  THIS IS YOUR RESEARCH CONTRIBUTION.")
# print("  Expected F1: 76-84%")
# print()
# print("  NOTE: This will take ~60-90 minutes on your GPU.")
# print()

# roberta_metrics, roberta_results = train_transformer(
#     model_name   = "roberta-base",
#     display_name = "RoBERTa-base",
#     save_dir     = "models/roberta_base"
# )
# all_results.append({
#     "model":            "RoBERTa-base",
#     "type":             "Proposed ★",
#     "test_metrics":     roberta_metrics,
#     "best_val_macro_f1":roberta_results["best_val_macro_f1"],
# })

# # =============================================================================
# # MODEL 4 — DistilBERT (efficient variant)
# # =============================================================================
# print(f"\n{'═'*65}")
# print("MODEL 4: DistilBERT-base-uncased (Efficient Variant)")
# print(f"{'═'*65}")
# print()
# print("  Why DistilBERT?")
# print("  Knowledge-distilled version of BERT (Sanh et al., 2019)")
# print("  40% smaller (66M params vs 110M), 60% faster inference")
# print("  Retains 97% of BERT performance on GLUE benchmark")
# print("  Important for deployment: web app needs fast inference")
# print("  Including this shows awareness of production constraints")
# print("  Expected F1: 70-78%")
# print()
# print("  NOTE: This will take ~40-60 minutes on your GPU.")
# print()

# distilbert_metrics, distilbert_results = train_transformer(
#     model_name   = "distilbert-base-uncased",
#     display_name = "DistilBERT-base-uncased",
#     save_dir     = "models/distilbert_base"
# )
# all_results.append({
#     "model":            "DistilBERT-base-uncased",
#     "type":             "Efficient Variant",
#     "test_metrics":     distilbert_metrics,
#     "best_val_macro_f1":distilbert_results["best_val_macro_f1"],
# })

# # =============================================================================
# FINAL COMPARISON TABLE — SCREENSHOT THIS
# =============================================================================
print(f"\n\n{'═'*65}")
print("FINAL COMPARISON TABLE — SCREENSHOT THIS")
print("PRIMARY METRIC: Macro-F1 Score")
print("Dataset: AMI Meeting Corpus (full)")
print(f"{'═'*65}")

rows = []
for r in all_results:
    m = r["test_metrics"]
    rows.append({
        "Model":           r["model"],
        "Type":            r["type"],
        "Macro-F1 (%)":    round(m["macro_f1"]*100,  1),
        "Precision (%)":   round(m["macro_precision"]*100, 1),
        "Recall (%)":      round(m["macro_recall"]*100,    1),
        "Binary-F1 (%)":   round(m["binary_f1"]*100,  1),
    })

df_results = pd.DataFrame(rows)
df_results = df_results.sort_values("Macro-F1 (%)", ascending=False)
print()
print(df_results.to_string(index=False))

# Find best model
best_row = df_results.iloc[0]
print()
print(f"Best model   : {best_row['Model']}")
print(f"Best Macro-F1: {best_row['Macro-F1 (%)']:.1f}%")

# # Compute improvements
# lr_f1 = df_results[df_results["Model"]=="TF-IDF + Logistic Regression"]["Macro-F1 (%)"].values[0]
# rb_f1 = df_results[df_results["Model"]=="RoBERTa-base"]["Macro-F1 (%)"].values[0]
# bt_f1 = df_results[df_results["Model"]=="BERT-base-uncased"]["Macro-F1 (%)"].values[0]

# print()
# print("RoBERTa improvements:")
# print(f"  vs TF-IDF+LR baseline  : +{rb_f1 - lr_f1:.1f}% macro-F1")
# print(f"  vs BERT-base baseline  : +{rb_f1 - bt_f1:.1f}% macro-F1")

# Save results
df_results.to_csv("data/processed/results/model_comparison.csv", index=False)
with open("data/processed/results/all_model_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print()
print("=" * 65)
print("Day 3 COMPLETE!")
print()
print("Files saved:")
print("  models/tfidf_lr/           ← TF-IDF + LR model")
print("  models/bert_base/          ← BERT-base model")
print("  models/roberta_base/       ← RoBERTa model (proposed)")
print("  models/distilbert_base/    ← DistilBERT model")
print("  data/processed/results/model_comparison.csv")
print("  data/processed/results/all_model_results.json")
print()
print("Screenshot the comparison table above.")
print("This is Table 2 in your research paper.")
print("=" * 65)