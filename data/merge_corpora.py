# merge_corpora.py
# =================================================================
# PURPOSE:
#   Combines AMI train + OOD train (MRDA + MeetingBank) into one
#   merged training set.
#
# WHY MERGE?
#   AMI alone  → model learns AMI-specific phrasing shortcuts
#   OOD alone  → model misses AMI annotation style entirely
#   AMI + OOD  → model must generalise across meeting styles
#
# The merged set forces the model to learn what action items MEAN
# rather than what they LOOK LIKE in one specific corpus.
#
# VALIDATION SET:
#   Stays AMI-only so validation scores are comparable across
#   all experiments (same distribution as Day 3 BERT baseline).
# =================================================================

import pandas as pd

print("=" * 60)
print("Merging AMI + OOD Training Data")
print("=" * 60)

# Load AMI training data
ami_train = pd.read_csv(
    "data/processed/train_labeled.csv"
).dropna(subset=["text"])

# Load OOD training data
ood_train = pd.read_csv(
    "data/processed/icsi_train.csv"
).dropna(subset=["text"])

# Rename speaker_id → speaker if needed
if ("speaker_id" in ood_train.columns and
        "speaker" not in ood_train.columns):
    ood_train = ood_train.rename(
        columns={"speaker_id": "speaker"}
    )

print(f"\nBefore merging:")
print(f"  AMI  train : {len(ami_train):,} utterances | "
      f"AI rate: {ami_train['action_item'].mean()*100:.1f}%")
print(f"  OOD  train : {len(ood_train):,} utterances | "
      f"AI rate: {ood_train['action_item'].mean()*100:.1f}%")

# Keep only the columns needed for training
# Both datasets must have these three columns
KEEP_COLS = ["meeting_id", "text", "action_item"]

ami_subset = ami_train[KEEP_COLS].copy()
ood_subset = ood_train[KEEP_COLS].copy()

# Tag the source so we can analyse by source later
ami_subset["source"] = "ami"
ood_subset["source"] = "ood"

# Combine and shuffle
merged = pd.concat(
    [ami_subset, ood_subset],
    ignore_index=True
).sample(
    frac=1, random_state=42
).reset_index(drop=True)

merged.to_csv(
    "data/processed/train_merged.csv", index=False
)

print(f"\nMerged training set:")
print(f"  Total      : {len(merged):,} utterances")
print(f"  AMI        : {(merged['source']=='ami').sum():,}")
print(f"  OOD        : {(merged['source']=='ood').sum():,}")
print(f"  AI rate    : {merged['action_item'].mean()*100:.1f}%")
print(f"\nClass distribution:")
vc = merged["action_item"].value_counts()
print(f"  Not action item (0): {vc.get(0,0):,}")
print(f"  Action item     (1): {vc.get(1,0):,}")
print(f"  Ratio 0:1          : "
      f"1 : {vc.get(0,0)/max(vc.get(1,1),1):.1f}")

print(f"\nSaved: data/processed/train_merged.csv")
print("Phase B COMPLETE. Push to GitHub, then run Phase C.")