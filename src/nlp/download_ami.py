# download_ami.py
import os, json, warnings, logging
import pandas as pd
from collections import defaultdict

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["DATASETS_VERBOSITY"] = "error"

import datasets
datasets.utils.logging.set_verbosity_error()
from datasets import load_dataset, Audio as HFAudio

os.makedirs("data/raw/ami",   exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("=" * 65)
print("MeetingLens - Day 1: Downloading Full AMI Meeting Corpus")
print("=" * 65)
print()

# ── STEP 1: Download ─────────────────────────────────────────────
print("Downloading from HuggingFace...")
dataset = load_dataset("edinburghcstr/ami", "ihm")
print("Download complete!")
for s in dataset:
    print(f"  {s}: {len(dataset[s]):,} rows")
print()

# ── STEP 2: Drop audio column (Audio type causes torchcodec crash)
KEEP = ["meeting_id", "speaker_id", "text", "begin_time", "end_time"]
print("Dropping audio column...")
for split in list(dataset.keys()):
    cols = [c for c in KEEP if c in dataset[split].column_names]
    dataset[split] = dataset[split].select_columns(cols)
print(f"Columns kept: {dataset['train'].column_names}")
print()

# ── STEP 3: Inspect ──────────────────────────────────────────────
print("Inspecting data structure...")
row = dataset["train"][0]
print(f"  meeting_id : {row['meeting_id']}")
print(f"  speaker_id : {row['speaker_id']}")
print(f"  text       : {repr(row['text'][:60])}")
print(f"  begin_time : {row['begin_time']}")
print(f"  end_time   : {row['end_time']}")
print()

# ── STEP 4: Group text rows → utterances → meetings ──────────────
def group_into_meetings(split_data):
    """
    Each row is one spoken segment (already a phrase, not just one word).
    Group by meeting_id, sort by begin_time, then merge consecutive
    same-speaker segments into full utterances.
    """
    raw = defaultdict(list)
    for row in split_data:
        raw[row["meeting_id"]].append({
            "speaker":    row["speaker_id"],
            "text":       row["text"].strip(),
            "begin_time": float(row["begin_time"] or 0),
            "end_time":   float(row["end_time"]   or 0),
        })

    meetings = {}
    for mid, segments in raw.items():
        segments.sort(key=lambda x: x["begin_time"])

        mtype = ("ES" if mid.startswith("ES") else
                 "TS" if mid.startswith("TS") else
                 "IS" if mid.startswith("IS") else "OTHER")

        utterances  = []
        cur_spk     = None
        cur_texts   = []
        cur_start   = 0.0
        cur_end     = 0.0

        for seg in segments:
            if not seg["text"]:
                continue
            if seg["speaker"] == cur_spk:
                cur_texts.append(seg["text"])
                cur_end = seg["end_time"]
            else:
                if cur_texts and cur_spk:
                    full_text = " ".join(cur_texts)
                    utterances.append({
                        "speaker":    cur_spk,
                        "text":       full_text,
                        "word_count": len(full_text.split()),
                        "begin_time": round(cur_start, 3),
                        "end_time":   round(cur_end,   3),
                        "duration":   round(cur_end - cur_start, 3),
                    })
                cur_spk   = seg["speaker"]
                cur_texts = [seg["text"]]
                cur_start = seg["begin_time"]
                cur_end   = seg["end_time"]

        if cur_texts and cur_spk:
            full_text = " ".join(cur_texts)
            utterances.append({
                "speaker":    cur_spk,
                "text":       full_text,
                "word_count": len(full_text.split()),
                "begin_time": round(cur_start, 3),
                "end_time":   round(cur_end,   3),
                "duration":   round(cur_end - cur_start, 3),
            })

        if len(utterances) >= 10:
            meetings[mid] = {
                "meeting_id":   mid,
                "meeting_type": mtype,
                "utterances":   utterances,
            }
    return meetings

print("Grouping segments into utterances (5-10 min)...")
all_meetings = {}
for split in dataset:
    print(f"  Processing {split}...")
    m = group_into_meetings(dataset[split])
    all_meetings.update(m)
    print(f"  -> {len(m)} meetings found")

print(f"\nTotal meetings: {len(all_meetings)}")

# ── STEP 5: Save JSON files ──────────────────────────────────────
print("\nSaving to data/raw/ami/...")
stats = []
for mid, meeting in all_meetings.items():
    utts        = meeting["utterances"]
    n_utt       = len(utts)
    n_spk       = len(set(u["speaker"] for u in utts))
    total_words = sum(u["word_count"] for u in utts)
    meeting.update({"n_utterances": n_utt, "n_speakers": n_spk,
                    "total_words": total_words})
    with open(f"data/raw/ami/{mid}.json", "w", encoding="utf-8") as f:
        json.dump(meeting, f, indent=2, ensure_ascii=False)
    stats.append({
        "meeting_id":   mid,
        "meeting_type": meeting["meeting_type"],
        "n_utterances": n_utt,
        "n_speakers":   n_spk,
        "total_words":  total_words,
    })

print(f"Saved {len(stats)} meetings.")

# ── STEP 6: Statistics ───────────────────────────────────────────
df = pd.DataFrame(stats).sort_values("meeting_id")
df.to_csv("data/processed/ami_corpus_stats.csv", index=False)

print()
print("=" * 65)
print("AMI CORPUS STATISTICS — SCREENSHOT THIS")
print("=" * 65)
print(f"\nTotal meetings saved    : {len(df)}")
print(f"Total utterances        : {df['n_utterances'].sum():,}")
print(f"Total words             : {df['total_words'].sum():,}")

print("\nBreakdown by meeting type:")
for mtype, g in df.groupby("meeting_type"):
    print(f"  {mtype}: {len(g):3d} meetings | "
          f"avg {g['n_utterances'].mean():.0f} utterances | "
          f"avg {g['n_speakers'].mean():.1f} speakers")

print("\nPer-meeting statistics:")
print(f"  Avg utterances / meeting : {df['n_utterances'].mean():.1f}")
print(f"  Avg words / meeting      : {df['total_words'].mean():.1f}")
print(f"  Avg speakers / meeting   : {df['n_speakers'].mean():.1f}")
print(f"  Min utterances           : {df['n_utterances'].min()}")
print(f"  Max utterances           : {df['n_utterances'].max()}")

print("\n5 largest meetings:")
print(df.nlargest(5, "n_utterances")[
    ["meeting_id","meeting_type","n_utterances","n_speakers","total_words"]
].to_string(index=False))

print("\n5 smallest meetings:")
print(df.nsmallest(5, "n_utterances")[
    ["meeting_id","meeting_type","n_utterances","n_speakers","total_words"]
].to_string(index=False))

print()
print("=" * 65)
print("Day 1 COMPLETE — Screenshot the table above!")
print("Files: data/raw/ami/  and  data/processed/")
print("=" * 65)