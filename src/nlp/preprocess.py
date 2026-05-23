# preprocess.py
# =============================================================================
# PURPOSE:
#   Reads all 171 raw AMI meeting JSON files, computes speaker statistics,
#   computes Speaker Dominance Score (SDS), and splits all meetings into
#   train / validation / test sets using STRATIFIED splitting.
#
# KEY CONCEPT — STRATIFIED SPLITTING:
#   The AMI corpus has 3 meeting types: ES, IS, TS.
#   If we split randomly, we might get:
#       train: all ES + some IS
#       test:  all TS + some IS
#   This means the model is tested on meeting styles it never trained on.
#   That is NOT a fair evaluation.
#
#   Stratified splitting fixes this by ensuring each split has the same
#   proportion of each type. If the corpus is 50% ES, 30% TS, 20% IS,
#   then train, val, and test each have roughly 50% ES, 30% TS, 20% IS.
#
# SPLIT SIZES:
#   70% train      → ~119 meetings → model learns from these
#   10% validation → ~17 meetings  → tune hyperparameters, early stopping
#   20% test       → ~35 meetings  → final evaluation ONLY (never touch during training)
#
# KEY CONCEPT — MEETING-LEVEL vs UTTERANCE-LEVEL SPLIT:
#   We split by MEETING, not by utterance.
#   Wrong way: put utterance 1-100 in train, utterance 101-200 in test
#              from the SAME meeting.
#   Problem:   The model sees utterance 99 in training. Utterance 101
#              is from the same conversation, so the model already knows
#              the context. This is called DATA LEAKAGE — it inflates
#              test scores artificially.
#   Right way: Put the ENTIRE meeting in either train or test.
#              The model never sees any utterance from a test meeting
#              during training.
#
# SPEAKER DOMINANCE SCORE (SDS):
#   SDS = max_speaker_talk_percentage / average_talk_percentage
#   Derived from inequality measurement literature (similar to Gini index).
#   SDS = 1.0 → everyone talks equally
#   SDS > 1.5 → one speaker dominates significantly
#   SDS > 2.0 → severe communication imbalance
#   Example: 4 speakers, one talks 60%, others share 40%.
#            Average = 25%. SDS = 60/25 = 2.4 → severe imbalance.
# =============================================================================

import json
import os
import random
import pandas as pd
from collections import defaultdict

# Fix random seed for reproducibility
# This means every time you run this script, you get the EXACT same split.
# Essential for research — reviewers must be able to reproduce your results.
random.seed(42)

print("=" * 65)
print("MeetingLens — Day 2 Step 1: Preprocessing Full AMI Corpus")
print("=" * 65)

# ── Load all raw meeting files ────────────────────────────────────────────────

RAW_DIR = "data/raw/ami"
all_files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".json")])
print(f"\nFound {len(all_files)} raw meeting files")

# Load each file and group by meeting type
meetings_by_type = defaultdict(list)

for fname in all_files:
    fpath = os.path.join(RAW_DIR, fname)
    with open(fpath, encoding="utf-8") as f:
        meeting = json.load(f)

    mtype = meeting.get("meeting_type", "OTHER")
    meetings_by_type[mtype].append(meeting)

print("\nMeetings by type:")
for mtype, mlist in sorted(meetings_by_type.items()):
    print(f"  {mtype}: {len(mlist)} meetings")

# ── Stratified split ──────────────────────────────────────────────────────────

train_meetings = []
val_meetings   = []
test_meetings  = []

for mtype, mlist in meetings_by_type.items():
    # Shuffle within each type before splitting
    # random.seed(42) above ensures this is reproducible
    random.shuffle(mlist)

    n       = len(mlist)
    n_test  = max(1, int(round(n * 0.20)))  # 20% for test
    n_val   = max(1, int(round(n * 0.10)))  # 10% for validation
    n_train = n - n_test - n_val            # remaining for train

    train_meetings += mlist[:n_train]
    val_meetings   += mlist[n_train : n_train + n_val]
    test_meetings  += mlist[n_train + n_val:]

    print(f"\n  {mtype} split:"
          f" train={n_train}, val={n_val}, test={n_test}")

print(f"\nFinal split sizes:")
print(f"  Train      : {len(train_meetings)} meetings")
print(f"  Validation : {len(val_meetings)} meetings")
print(f"  Test       : {len(test_meetings)} meetings")
print(f"  Total      : {len(train_meetings)+len(val_meetings)+len(test_meetings)} meetings")

# ── Helper function — compute speaker statistics ──────────────────────────────

def compute_speaker_stats(utterances):
    """
    Computes per-speaker word count, utterance count, and talk percentage.
    Also computes the Speaker Dominance Score (SDS) for the whole meeting.

    Parameters:
        utterances (list): list of utterance dicts with 'speaker' and 'word_count'

    Returns:
        stats (dict): {speaker_id: {word_count, utterance_count, talk_pct}}
        sds   (float): Speaker Dominance Score for the meeting
    """
    word_counts = defaultdict(int)
    utt_counts  = defaultdict(int)

    for utt in utterances:
        spk = utt["speaker"]
        wc  = utt.get("word_count", len(utt["text"].split()))
        word_counts[spk] += wc
        utt_counts[spk]  += 1

    total_words = sum(word_counts.values())
    if total_words == 0:
        return {}, 1.0

    # Build stats dict
    stats = {}
    for spk in word_counts:
        stats[spk] = {
            "word_count":      word_counts[spk],
            "utterance_count": utt_counts[spk],
            "talk_pct":        round(word_counts[spk] / total_words * 100, 4)
        }

    # Compute SDS
    # SDS = max_talk_pct / average_talk_pct
    talk_pcts = [s["talk_pct"] for s in stats.values()]
    avg_pct   = sum(talk_pcts) / len(talk_pcts)
    max_pct   = max(talk_pcts)
    sds       = round(max_pct / avg_pct, 4) if avg_pct > 0 else 1.0

    return stats, sds

# ── Process and save each split ───────────────────────────────────────────────

for split_name, split_meetings in [
    ("train",      train_meetings),
    ("validation", val_meetings),
    ("test",       test_meetings),
]:
    out_dir = f"data/processed/{split_name}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nProcessing {split_name} split ({len(split_meetings)} meetings)...")
    summaries = []

    for meeting in split_meetings:
        mid   = meeting["meeting_id"]
        mtype = meeting["meeting_type"]

        # Filter empty utterances
        utterances = [u for u in meeting["utterances"]
                      if u.get("text", "").strip()]

        # Ensure word_count is set on every utterance
        for u in utterances:
            u["word_count"] = len(u["text"].split())

        # Compute speaker statistics and SDS
        speaker_stats, sds = compute_speaker_stats(utterances)

        # Build the processed meeting record
        processed = {
            "meeting_id":              mid,
            "meeting_type":            mtype,
            "split":                   split_name,
            "total_utterances":        len(utterances),
            "total_words":             sum(u["word_count"] for u in utterances),
            "n_speakers":              len(speaker_stats),
            "speaker_stats":           speaker_stats,
            "speaker_dominance_score": sds,
            "dominant_speaker_flag":   1 if sds > 1.5 else 0,
            "utterances":              utterances,
        }

        # Save to split folder
        out_path = os.path.join(out_dir, f"{mid}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False)

        summaries.append({
            "meeting_id":              mid,
            "meeting_type":            mtype,
            "split":                   split_name,
            "n_utterances":            len(utterances),
            "n_speakers":              len(speaker_stats),
            "sds":                     sds,
            "dominant_speaker_flag":   1 if sds > 1.5 else 0,
            "total_words":             processed["total_words"],
        })

    # Save summary CSV for this split
    df_split = pd.DataFrame(summaries)
    df_split.to_csv(
        f"data/processed/{split_name}_meeting_summary.csv",
        index=False
    )

    # Print split stats
    print(f"  Saved {len(summaries)} meetings")
    print(f"  Avg utterances / meeting : {df_split['n_utterances'].mean():.1f}")
    print(f"  Avg speakers / meeting   : {df_split['n_speakers'].mean():.1f}")
    print(f"  Avg SDS                  : {df_split['sds'].mean():.3f}")
    print(f"  Meetings with SDS > 1.5  : {df_split['dominant_speaker_flag'].sum()}")
    print(f"  Meeting types            : {df_split['meeting_type'].value_counts().to_dict()}")

# ── Combined summary ──────────────────────────────────────────────────────────

all_summaries = []
for split_name in ["train", "validation", "test"]:
    csv_path = f"data/processed/{split_name}_meeting_summary.csv"
    df_s = pd.read_csv(csv_path)
    all_summaries.append(df_s)

df_all = pd.concat(all_summaries, ignore_index=True)
df_all.to_csv("data/processed/_all_meetings_summary.csv", index=False)

print("\n" + "=" * 65)
print("PREPROCESSING COMPLETE — FULL AMI CORPUS PROCESSED AND SPLIT")
print("=" * 65)

summary_rows = []
for split_name in ["train", "validation", "test"]:
    sub = df_all[df_all["split"] == split_name]
    summary_rows.append({
        "Split":        split_name,
        "Meetings":     len(sub),
        "Utterances":   sub["n_utterances"].sum(),
        "Avg Utt/Mtg":  round(sub["n_utterances"].mean(), 1),
        "Avg Spk/Mtg":  round(sub["n_speakers"].mean(), 1),
        "Avg SDS":      round(sub["sds"].mean(), 3),
        "Dominant Spkr": sub["dominant_speaker_flag"].sum(),
    })

df_summary = pd.DataFrame(summary_rows)
print()
print(df_summary.to_string(index=False))
print()
print(f"Total utterances across all splits : {df_all['n_utterances'].sum():,}")
print(f"Total words across all splits      : {df_all['total_words'].sum():,}")
print()
print("Files saved to data/processed/")
print("Day 2 Step 1 COMPLETE!")