# src/nlp/download_icsi.py
# Fixed version based on actual MeetingBank column structure
# 'source' column contains the dialogue transcript
# 'meeting_id' column exists already

import os, re, warnings, logging
import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["DATASETS_VERBOSITY"] = "error"

import datasets
datasets.utils.logging.set_verbosity_error()
from datasets import load_dataset

os.makedirs("data/raw/icsi",  exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("=" * 65)
print("MeetingLens - Downloading MeetingBank (OOD corpus)")
print("=" * 65)

# ── Load MeetingBank ──────────────────────────────────────────
print("\nLoading MeetingBank...")
mb = load_dataset("lytang/MeetingBank-transcript")

first_split = list(mb.keys())[0]
print(f"Splits  : {list(mb.keys())}")
print(f"Columns : {mb[first_split].column_names}")
print(f"Rows    : {len(mb[first_split]):,}")

# Print first row fully so we can verify
print("\nFirst row (full):")
row0 = mb[first_split][0]
for k, v in row0.items():
    print(f"  {k:<15}: {str(v)[:200]}")

# ── The transcript is in 'source' column ─────────────────────
# Format is: "Speaker 0: text\nSpeaker 1: text\n..."
# meeting_id column already exists

print("\nParsing transcripts from 'source' column...")

ACTION_PATTERNS = [
    r"\b(will|shall)\s+\w+",
    r"\b(need to|needs to|have to|has to)\b",
    r"\b(going to|gonna)\b",
    r"\bcan you\b",
    r"\bwould you\b",
    r"\bcould you\b",
    r"\bplease\s+\w+",
    r"\bshould\s+(we|you|I|he|she|they)\b",
    r"\b(make sure|ensure|action item|follow.?up)\b",
    r"\bmove(d)?\s+(to|that)\b",
    r"\bI so move\b",
    r"\bmotion\s+(to|that)\b",
    r"\brecommend(s|ed)?\b",
]
NEGATIVE_PATTERNS = [
    r"\b(used to|supposed to)\b",
    r"\byesterday|last week|last month|last year\b",
    r"\balready\b",
    r"\b(should have|would have|could have)\b",
    r"\bwill\s+(probably|maybe|perhaps)\b",
    r"\b(will this|will it|will that)\b",
    r"\bin theory|ideally|hypothetically\b",
]
compiled_pos = [re.compile(p, re.IGNORECASE) for p in ACTION_PATTERNS]
compiled_neg = [re.compile(p, re.IGNORECASE) for p in NEGATIVE_PATTERNS]

def label_text(text):
    for neg in compiled_neg:
        if neg.search(text):
            return 0
    for pos in compiled_pos:
        if pos.search(text):
            return 1
    return 0

all_rows = []

for split in mb:
    print(f"  Processing split: {split} ({len(mb[split]):,} rows)...")
    for i, row in enumerate(mb[split]):

        meeting_id = str(row["meeting_id"]) if row.get("meeting_id") else f"MB_{i:05d}"

        # The transcript is in 'source' column
        dialogue = str(row["source"]) if row.get("source") else ""

        if not dialogue or len(dialogue.strip()) < 10:
            continue

        # Split into individual speaker turns
        # Format: "Speaker 0: text\nSpeaker 1: text\n..."
        lines = dialogue.strip().split("\n")

        for j, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 5:
                continue

            # Parse "Speaker X: actual text"
            if ":" in line:
                parts   = line.split(":", 1)
                speaker = parts[0].strip()
                text    = parts[1].strip() if len(parts) > 1 else ""
            else:
                speaker = "UNK"
                text    = line

            # Skip very short utterances
            if not text or len(text.split()) < 3:
                continue

            all_rows.append({
                "meeting_id":  f"MB_{meeting_id}",
                "speaker_id":  speaker,
                "text":        text,
                "action_item": label_text(text),
                "begin_time":  float(j),
                "end_time":    float(j + 1),
                "source":      "meetingbank",
            })

# ── Build DataFrame ───────────────────────────────────────────
df = pd.DataFrame(all_rows)

if df.empty:
    print("\nERROR: No rows extracted. Check column names above.")
    exit(1)

print(f"\nMeetingBank parsed:")
print(f"  Total utterances : {len(df):,}")
print(f"  Total meetings   : {df['meeting_id'].nunique()}")
print(f"  Action items     : {df['action_item'].sum():,} "
      f"({df['action_item'].mean()*100:.1f}%)")

# Sanity check — show sample utterances
print("\nSample ACTION utterances (verify these look right):")
actions = df[df["action_item"] == 1]
if len(actions) >= 5:
    for t in actions["text"].sample(5, random_state=42):
        print(f"  ACTION: {t[:90]}")
else:
    print(f"  Only {len(actions)} action items found — check labeling rules")

print("\nSample NON-ACTION utterances:")
non = df[df["action_item"] == 0]
for t in non["text"].sample(min(3, len(non)), random_state=42):
    print(f"  OTHER:  {t[:90]}")

# ── Save ──────────────────────────────────────────────────────
df.to_csv("data/processed/icsi_labeled.csv", index=False)
df.to_csv("data/processed/meetingbank_labeled.csv", index=False)

print()
print("=" * 65)
print("COMPLETE")
print("=" * 65)
print(f"Saved: data/processed/icsi_labeled.csv")
print(f"Saved: data/processed/meetingbank_labeled.csv")
print(f"\nNext step: python src/nlp/split_icsi.py")