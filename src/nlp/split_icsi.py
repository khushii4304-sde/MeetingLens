# src/nlp/split_icsi.py
# Splits MeetingBank (our OOD corpus) into train and test sets
# Split is done at MEETING level — all utterances from one meeting
# go entirely to train OR entirely to test, never both.
#
# WHY MEETING-LEVEL SPLIT:
# If we split by utterance randomly, the same meeting could appear
# in both train and test. The model would see context from the same
# conversation in both sets — that is data leakage.
# Meeting-level split ensures the model never sees any utterance
# from a test meeting during training.

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

print("=" * 65)
print("Splitting MeetingBank into train / test (meeting-level)")
print("=" * 65)

# ── Load ──────────────────────────────────────────────────────
df = pd.read_csv("data/processed/icsi_labeled.csv")
print(f"\nLoaded: {len(df):,} utterances")
print(f"Meetings   : {df['meeting_id'].nunique()}")
print(f"Action rate: {df['action_item'].mean()*100:.1f}%")

# ── Meeting-level split ───────────────────────────────────────
# groups = meeting_id column tells GroupShuffleSplit to keep
# all rows from one meeting together
meetings = df["meeting_id"].values

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.25,   # 25% of MEETINGS go to test
    random_state=42
)
tr_idx, te_idx = next(gss.split(df, groups=meetings))

icsi_train = df.iloc[tr_idx].reset_index(drop=True)
icsi_test  = df.iloc[te_idx].reset_index(drop=True)

# ── Leakage check ─────────────────────────────────────────────
train_meetings = set(icsi_train["meeting_id"].unique())
test_meetings  = set(icsi_test["meeting_id"].unique())
overlap        = train_meetings & test_meetings

print(f"\nLeakage check:")
print(f"  Train meetings : {len(train_meetings)}")
print(f"  Test meetings  : {len(test_meetings)}")
print(f"  Overlap        : {len(overlap)}  ← must be 0")

if overlap:
    print(f"ERROR: Meeting leakage detected: {overlap}")
    exit(1)
else:
    print("  PASSED — zero overlap")

# ── Stats ─────────────────────────────────────────────────────
print(f"\nTrain split:")
print(f"  Utterances  : {len(icsi_train):,}")
print(f"  Meetings    : {icsi_train['meeting_id'].nunique()}")
print(f"  Action items: {icsi_train['action_item'].sum():,} "
      f"({icsi_train['action_item'].mean()*100:.1f}%)")

print(f"\nTest split:")
print(f"  Utterances  : {len(icsi_test):,}")
print(f"  Meetings    : {icsi_test['meeting_id'].nunique()}")
print(f"  Action items: {icsi_test['action_item'].sum():,} "
      f"({icsi_test['action_item'].mean()*100:.1f}%)")

# ── Save ──────────────────────────────────────────────────────
icsi_train.to_csv("data/processed/icsi_train.csv", index=False)
icsi_test.to_csv( "data/processed/icsi_test.csv",  index=False)

print(f"\nSaved:")
print(f"  data/processed/icsi_train.csv")
print(f"  data/processed/icsi_test.csv")
print()
print("=" * 65)
print("Split complete.")
print("Next steps in order:")
print("  1. python src/nlp/merge_corpora.py")
print("  2. python src/nlp/evaluate_zeroshot.py")
print("  3. python src/nlp/train_model.py  (RoBERTa + DistilBERT)")
print("=" * 65)