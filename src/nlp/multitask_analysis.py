# src/nlp/multitask_analysis.py
# Adds three additional analytical dimensions beyond action item detection:
# 1. Decision Detection
# 2. Speaker Dominance Analysis  
# 3. Sentiment / Meeting Tone Analysis
#
# These use rule-based approaches which is standard and legitimate
# for secondary analytics tasks in meeting intelligence systems.
# The PRIMARY task (action item extraction) uses trained ML models.
# Secondary tasks use linguistic rules — this is how industry tools
# like Otter.ai and Fireflies work too.

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import re
import json
import pandas as pd
import numpy as np
from collections import Counter

os.makedirs("data/processed/results", exist_ok=True)

print("=" * 65)
print("Multi-Task Meeting Intelligence Analysis")
print("=" * 65)

# ── Load data ─────────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_csv("data/processed/train_labeled.csv").dropna(subset=["text"])
test  = pd.read_csv("data/processed/test_labeled.csv").dropna(subset=["text"])
df    = pd.concat([train, test], ignore_index=True)

print(f"Total utterances : {len(df):,}")
print(f"Total meetings   : {df['meeting_id'].nunique()}")
print(f"Action item rate : {df['action_item'].mean()*100:.1f}%")

# =============================================================
# TASK 1 — DECISION DETECTION
# =============================================================
# Decisions are moments where the group reaches a conclusion.
# Different from action items:
#   Action item = "John will send the report" (future task)
#   Decision    = "We agreed to use the new design" (conclusion reached)
#
# We use linguistic patterns specific to decision language.
# Research shows decisions cluster around agreement markers,
# resolution language, and past-tense commitment phrases.

print("\n" + "=" * 65)
print("TASK 1: Decision Detection")
print("=" * 65)

DECISION_PATTERNS = [
    # Agreement / resolution
    r"\bwe (have )?(decided|agreed|resolved|concluded|determined)\b",
    r"\bthe (group|team|committee) (has )?(decided|agreed|resolved)\b",
    r"\bfinal(ly)? (decided|agreed|chose|selected|approved|adopted)\b",

    # Formal council/meeting language
    r"\bmotion (carried|passed|approved|adopted|failed)\b",
    r"\ball (those )?(in favor|agreed|ayes)\b",
    r"\bunanimously (approved|adopted|agreed|passed)\b",
    r"\bso (ordered|resolved|agreed)\b",
    r"\bthe (ayes|nays) have it\b",

    # Conclusion markers
    r"\bwe('re| are) going (to go )?with\b",
    r"\bwe('ve| have) (chosen|selected|picked|decided on)\b",
    r"\bthe decision is\b",
    r"\bit('s| is) (been |)decided\b",
    r"\bwe('ll| will) go (ahead )?with\b",
    r"\bthat('s| is) (settled|decided|confirmed|agreed)\b",

    # Voting outcomes
    r"\b(passes|passed) (unanimously|with|by)\b",
    r"\bvote (is |was )?(unanimous|carried|passed|failed)\b",
    r"\b\d+ (to|ayes|votes) (to |for |against )?\d+\b",
]

DECISION_NEGATIVES = [
    r"\bwe (haven't|have not|didn't|did not) (decide|agree|resolve)\b",
    r"\bnot (yet |)decided\b",
    r"\bstill (need to |)decide\b",
    r"\bno (decision|agreement|consensus)\b",
    r"\bundecided\b",
]

compiled_dec_pos = [re.compile(p, re.IGNORECASE) for p in DECISION_PATTERNS]
compiled_dec_neg = [re.compile(p, re.IGNORECASE) for p in DECISION_NEGATIVES]

def detect_decision(text):
    text = str(text)
    for neg in compiled_dec_neg:
        if neg.search(text):
            return 0
    for pos in compiled_dec_pos:
        if pos.search(text):
            return 1
    return 0

df["is_decision"] = df["text"].apply(detect_decision)

dec_count = df["is_decision"].sum()
dec_rate  = df["is_decision"].mean() * 100

print(f"\nResults:")
print(f"  Total utterances   : {len(df):,}")
print(f"  Decisions detected : {dec_count:,} ({dec_rate:.1f}%)")

# How often do action items and decisions co-occur?
both = ((df["action_item"] == 1) & (df["is_decision"] == 1)).sum()
print(f"  Both action+decision: {both:,}")
print(f"  Action items without decisions: "
      f"{((df['action_item']==1) & (df['is_decision']==0)).sum():,}")
print(f"  Decisions without action items: "
      f"{((df['action_item']==0) & (df['is_decision']==1)).sum():,}")

print(f"\nSample detected decisions:")
samples = df[df["is_decision"]==1]["text"].sample(
    min(8, dec_count), random_state=42
).tolist()
for s in samples:
    print(f"  DECISION: {s[:90]}")

# Decision rate by meeting type
if "meeting_id" in df.columns:
    df["meeting_type"] = df["meeting_id"].apply(
        lambda x: "ES" if str(x).startswith("ES") else
                  "TS" if str(x).startswith("TS") else
                  "IS" if str(x).startswith("IS") else "OTHER"
    )
    dec_by_type = df.groupby("meeting_type")["is_decision"].agg(
        ["sum", "mean", "count"]
    ).rename(columns={"sum":"decisions","mean":"rate","count":"utterances"})
    dec_by_type["rate"] = (dec_by_type["rate"] * 100).round(1)
    print(f"\nDecision rate by meeting type:")
    print(dec_by_type.to_string())

# =============================================================
# TASK 2 — SPEAKER DOMINANCE ANALYSIS
# =============================================================
# Speaker dominance measures participation inequality.
# In meeting intelligence, this is important because:
# - Dominant speakers tend to assign more action items
# - Quiet speakers may be disengaged or marginalized
# - Balanced participation correlates with better meeting outcomes
#
# We measure dominance using:
# 1. Utterance share (% of total turns)
# 2. Word share (% of total words spoken)
# 3. Action item ownership (who assigns vs who receives tasks)
# 4. Gini coefficient (standard inequality measure)

print("\n" + "=" * 65)
print("TASK 2: Speaker Dominance Analysis")
print("=" * 65)

if "speaker_id" not in df.columns:
    print("speaker_id column not found — skipping speaker analysis")
else:
    df["word_count"] = df["text"].str.split().str.len()

    speaker_stats = df.groupby("speaker_id").agg(
        utterances    = ("text",        "count"),
        words         = ("word_count",  "sum"),
        action_items  = ("action_item", "sum"),
        decisions     = ("is_decision", "sum"),
    ).reset_index()

    total_utt   = speaker_stats["utterances"].sum()
    total_words = speaker_stats["words"].sum()

    speaker_stats["utt_share_%"]  = (
        speaker_stats["utterances"] / total_utt * 100
    ).round(1)
    speaker_stats["word_share_%"] = (
        speaker_stats["words"] / total_words * 100
    ).round(1)
    speaker_stats["ai_rate_%"]    = (
        speaker_stats["action_items"] / speaker_stats["utterances"] * 100
    ).round(1)

    speaker_stats = speaker_stats.sort_values(
        "utterances", ascending=False
    ).reset_index(drop=True)

    print(f"\nTotal unique speakers: {len(speaker_stats)}")
    print(f"\nTop 10 speakers by utterance count:")
    print(speaker_stats.head(10)[[
        "speaker_id","utterances","utt_share_%",
        "words","word_share_%","action_items","ai_rate_%"
    ]].to_string(index=False))

    # Gini coefficient — standard inequality measure
    # 0 = perfectly equal, 1 = one person talks entirely
    shares = speaker_stats["utterances"].values.astype(float)
    shares = shares / shares.sum()
    shares = np.sort(shares)
    n      = len(shares)
    cumsum = np.cumsum(shares)
    gini   = (n + 1 - 2 * cumsum.sum() / cumsum[-1]) / n

    print(f"\nSpeaker dominance metrics:")
    print(f"  Total speakers        : {len(speaker_stats)}")
    print(f"  Top speaker share     : {speaker_stats['utt_share_%'].iloc[0]:.1f}%")
    print(f"  Top 3 speakers share  : {speaker_stats['utt_share_%'].head(3).sum():.1f}%")
    print(f"  Gini coefficient      : {gini:.3f}")
    print(f"  Interpretation        : ", end="")
    if gini < 0.3:
        print("Low inequality — balanced participation")
    elif gini < 0.5:
        print("Moderate inequality — some dominance")
    else:
        print("High inequality — strongly dominated by few speakers")

    # Who assigns vs who receives action items?
    top_ai_speakers = speaker_stats.nlargest(5, "action_items")[
        ["speaker_id", "action_items", "ai_rate_%", "utterances"]
    ]
    print(f"\nTop 5 action item assigners:")
    print(top_ai_speakers.to_string(index=False))

    # Save speaker stats
    speaker_stats.to_csv(
        "data/processed/results/speaker_dominance.csv", index=False
    )
    print(f"\nSaved: data/processed/results/speaker_dominance.csv")

# =============================================================
# TASK 3 — SENTIMENT / MEETING TONE ANALYSIS
# =============================================================
# Meeting tone affects productivity and decision quality.
# We classify utterances into positive, negative, and neutral
# using a lexicon-based approach.
#
# Why rule-based and not a trained model?
# 1. Meeting language is domain-specific — generic sentiment
#    models (trained on reviews/tweets) perform poorly on it
# 2. Rule-based is interpretable and auditable
# 3. Sufficient for the analytical purpose here
#
# Key research question: do action items cluster in
# positive or negative moments of meetings?

print("\n" + "=" * 65)
print("TASK 3: Sentiment and Meeting Tone Analysis")
print("=" * 65)

# Meeting-specific sentiment lexicons
POSITIVE_LEXICON = [
    "great", "excellent", "good", "agree", "perfect", "thanks",
    "appreciate", "wonderful", "yes", "absolutely", "fantastic",
    "brilliant", "happy", "glad", "pleased", "support", "love",
    "impressive", "well done", "congratulations", "helpful",
    "positive", "right", "correct", "success", "successful",
    "approved", "accepted", "achieved", "accomplished", "done",
]

NEGATIVE_LEXICON = [
    "problem", "issue", "concern", "disagree", "wrong", "no",
    "cannot", "difficult", "unfortunately", "failed", "failure",
    "bad", "poor", "terrible", "worried", "worried", "stuck",
    "blocked", "delayed", "behind", "missed", "error", "bug",
    "broken", "unclear", "confused", "frustrat", "disappoint",
    "reject", "denied", "impossible", "risk", "threat", "crisis",
]

def get_sentiment(text):
    text_lower = str(text).lower()
    pos = sum(1 for w in POSITIVE_LEXICON if w in text_lower)
    neg = sum(1 for w in NEGATIVE_LEXICON if w in text_lower)
    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    else:
        return "neutral"

df["sentiment"] = df["text"].apply(get_sentiment)

sent_dist = df["sentiment"].value_counts()
print(f"\nOverall sentiment distribution:")
for sent, count in sent_dist.items():
    pct = count / len(df) * 100
    print(f"  {sent:<10}: {count:,} ({pct:.1f}%)")

# Key research insight: do action items cluster in positive moments?
print(f"\nAction item rate by sentiment:")
sent_ai = df.groupby("sentiment")["action_item"].agg(
    ["sum", "mean", "count"]
).rename(columns={"sum":"action_items","mean":"rate","count":"utterances"})
sent_ai["rate_%"] = (sent_ai["rate"] * 100).round(1)
print(sent_ai[["utterances","action_items","rate_%"]].to_string())

print(f"\nDecision rate by sentiment:")
sent_dec = df.groupby("sentiment")["is_decision"].agg(
    ["sum", "mean", "count"]
).rename(columns={"sum":"decisions","mean":"rate","count":"utterances"})
sent_dec["rate_%"] = (sent_dec["rate"] * 100).round(1)
print(sent_dec[["utterances","decisions","rate_%"]].to_string())

# Sentiment progression — does meeting tone change over time?
# Use utterance index as proxy for time
df_sorted = df.sort_values(["meeting_id", "begin_time"] 
                            if "begin_time" in df.columns 
                            else ["meeting_id"]).reset_index(drop=True)
df_sorted["position"] = df_sorted.groupby("meeting_id").cumcount()
df_sorted["pos_pct"]  = df_sorted.groupby("meeting_id")["position"].transform(
    lambda x: x / x.max() if x.max() > 0 else 0
)

# Divide meeting into thirds
df_sorted["meeting_third"] = pd.cut(
    df_sorted["pos_pct"],
    bins=[0, 0.33, 0.67, 1.0],
    labels=["Opening", "Middle", "Closing"],
    include_lowest=True
)

print(f"\nSentiment by meeting phase:")
phase_sent = df_sorted.groupby(
    ["meeting_third", "sentiment"], observed=True
).size().unstack(fill_value=0)
phase_pct  = phase_sent.div(phase_sent.sum(axis=1), axis=0) * 100
print(phase_pct.round(1).to_string())

print(f"\nAction item rate by meeting phase:")
phase_ai = df_sorted.groupby("meeting_third", observed=True)["action_item"].agg(
    ["sum", "mean", "count"]
)
phase_ai["rate_%"] = (phase_ai["mean"] * 100).round(1)
print(phase_ai[["count","sum","rate_%"]].to_string())

print(f"\nInsight: Action items are most common in the")
closing_rate = phase_ai.loc["Closing","rate_%"] if "Closing" in phase_ai.index else 0
opening_rate = phase_ai.loc["Opening","rate_%"] if "Opening" in phase_ai.index else 0
if closing_rate > opening_rate:
    print(f"  CLOSING phase ({closing_rate:.1f}%) vs opening ({opening_rate:.1f}%)")
    print(f"  Meetings assign tasks toward the end — expected.")
else:
    print(f"  OPENING phase ({opening_rate:.1f}%) vs closing ({closing_rate:.1f}%)")

# =============================================================
# SAVE ALL RESULTS
# =============================================================
results = {
    "decision_detection": {
        "total_utterances":   int(len(df)),
        "decisions_detected": int(dec_count),
        "decision_rate_%":    round(dec_rate, 2),
        "co_occurrence_with_actions": int(both),
    },
    "sentiment_analysis": {
        "positive_%": round(
            sent_dist.get("positive",0)/len(df)*100, 2),
        "neutral_%":  round(
            sent_dist.get("neutral",0)/len(df)*100, 2),
        "negative_%": round(
            sent_dist.get("negative",0)/len(df)*100, 2),
        "action_item_rate_by_sentiment": {
            s: round(r*100,2)
            for s,r in df.groupby("sentiment")["action_item"].mean().items()
        }
    },
}

with open("data/processed/results/multitask_results.json", "w") as f:
    json.dump(results, f, indent=2)

df.to_csv("data/processed/multitask_labeled.csv", index=False)

print("\n" + "=" * 65)
print("Multi-Task Analysis COMPLETE")
print("=" * 65)
print("Files saved:")
print("  data/processed/results/multitask_results.json")
print("  data/processed/results/speaker_dominance.csv")
print("  data/processed/multitask_labeled.csv")
print("\nNext: python src/nlp/ablation_study.py")