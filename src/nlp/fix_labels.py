# src/nlp/fix_labels_v2.py
# Balanced version — target 12-20% positive rate

import re
import pandas as pd

print("Loading icsi_labeled.csv...")
df = pd.read_csv("data/processed/icsi_labeled.csv")
print(f"Current: {len(df):,} utterances, {df['action_item'].mean()*100:.1f}% positive")

# ── BALANCED rules ────────────────────────────────────────────

POSITIVE_PATTERNS = [
    # Direct future commitment with action verb — "I/we will + action verb"
    r"\b(I|we)\s+will\s+(send|submit|prepare|review|provide|follow|"
    r"contact|schedule|update|complete|finish|draft|coordinate|report|"
    r"check|confirm|ensure|make|create|write|look into|reach out|"
    r"get back|present|share|distribute|bring|work|handle|take care|"
    r"move forward|proceed|address|respond|notify|inform|arrange|"
    r"set up|put together|pull|gather|verify|circulate|post)\b",

    # "You will" — direct task assignment
    r"\byou\s+will\s+(send|submit|prepare|review|provide|follow|"
    r"contact|schedule|update|complete|check|confirm|present|share)\b",

    # Can/could/would you + action verb (requests)
    r"\b(can|could|would)\s+you\s+(please\s+)?(send|submit|prepare|"
    r"review|provide|follow|contact|schedule|update|complete|check|"
    r"confirm|look into|get back|share|bring|handle|reach out|"
    r"coordinate|verify|draft|present|notify|arrange|set up)\b",

    # Need to / needs to / have to + action verb
    r"\b(need|needs|have|has)\s+to\s+(send|submit|prepare|review|"
    r"provide|follow|contact|schedule|update|complete|check|confirm|"
    r"look into|get back|share|bring|handle|reach out|coordinate|"
    r"verify|draft|present|notify|arrange|set up|make sure|ensure)\b",

    # Going to + action verb (commitment)
    r"\b(I|we|you)\s+(am|are|is|'m|'re)\s+going\s+to\s+(send|submit|"
    r"prepare|review|provide|follow|contact|schedule|update|complete|"
    r"check|confirm|look into|get back|share|bring|handle|"
    r"coordinate|verify|draft|present|notify|arrange|set up|make|"
    r"move|approve|adopt)\b",

    r"\b(I|we|you)\s+going\s+to\s+(send|submit|prepare|review|"
    r"provide|contact|schedule|update|complete|check|confirm|"
    r"look into|share|bring|handle|coordinate|verify|draft|"
    r"present|notify|approve|adopt|move|make)\b",

    # Formal motion language (council meetings)
    r"\bI\s+(so\s+)?move\b",
    r"\bI\s+move\s+(to|that)\b",
    r"\bmotion\s+to\s+\w+",
    r"\bmake\s+a\s+motion\s+to\b",
    r"\bI\s+second\b",
    r"\bI\s+second\s+that\b",

    # Recommendation language (very common in council meetings)
    r"\brecommendation\s+to\s+(approve|adopt|award|authorize|"
    r"increase|decrease|accept|reject|amend|establish|create|"
    r"declare|receive|direct|request|support|oppose)\b",

    r"\brecommend\s+(to|that)\s+\w+",

    # Action item / follow-up (explicit)
    r"\baction item\b",
    r"\bfollow[- ]?up\s+(with|on|by|to)\b",

    # Make sure / ensure + clause
    r"\bmake sure\s+(that\s+)?\w+",
    r"\bensure\s+(that\s+)?\w+",

    # Staff direction (common in council meetings)
    r"\bstaff\s+(will|should|is\s+directed|is\s+requested|needs?\s+to)\b",
    r"\bdirect\s+staff\s+to\b",
    r"\bask\s+staff\s+to\b",
    r"\brequest\s+that\s+staff\b",

    # Please + action verb (direct instruction)
    r"\bplease\s+(send|submit|prepare|review|provide|contact|"
    r"schedule|update|complete|check|confirm|look into|share|"
    r"bring|handle|coordinate|verify|draft|present|notify|"
    r"arrange|set up|make sure|follow up)\b",
]

NEGATIVE_PATTERNS = [
    # Negations — these flip meaning of modal verbs
    r"\bwill\s+not\b",
    r"\bwon'?t\b",
    r"\bwould\s+not\b",
    r"\bwouldn'?t\b",
    r"\bcould\s+not\b",
    r"\bcouldn'?t\b",
    r"\bshould\s+not\b",
    r"\bshouldn'?t\b",
    r"\bcannot\b",
    r"\bcan'?t\b",
    r"\bneed\s+not\b",
    r"\bneedn'?t\b",

    # Past tense — already done
    r"\b(already|completed|finished|done|submitted|sent|reviewed|"
    r"approved|adopted|awarded|authorized|accepted)\b",

    r"\b(yesterday|last\s+week|last\s+month|last\s+year|"
    r"last\s+time|previously|earlier\s+today|ago)\b",

    # Hypothetical past
    r"\b(should\s+have|would\s+have|could\s+have|might\s+have|"
    r"must\s+have|need\s+have)\b",

    # Clearly not task-related physical actions
    r"\b(turn\s+around|sit\s+down|stand\s+up|take\s+a\s+picture|"
    r"take\s+your\s+picture|put\s+your\s+hand|raise\s+your\s+hand)\b",

    # Pure questions (end with ? and start with question word)
    # Only block if it's ONLY a question with no commitment
    r"^(what|when|where|who|why|is\s+there|are\s+there|"
    r"does\s+anyone|did\s+anyone)\s.*\?$",
]

compiled_pos = [re.compile(p, re.IGNORECASE) for p in POSITIVE_PATTERNS]
compiled_neg = [re.compile(p, re.IGNORECASE) for p in NEGATIVE_PATTERNS]

def balanced_label(text):
    text = text.strip()

    # Skip extremely short utterances (1-2 words only)
    if len(text.split()) < 3:
        return 0

    # Negative patterns override everything
    for neg in compiled_neg:
        if neg.search(text):
            return 0

    # Check positive patterns
    for pos in compiled_pos:
        if pos.search(text):
            return 1

    return 0

print("Applying balanced labels...")
df["action_item"] = df["text"].apply(balanced_label)

rate = df['action_item'].mean() * 100
print(f"After : {len(df):,} utterances, {rate:.1f}% positive")
print(f"Action items: {df['action_item'].sum():,}")

# ── Detailed verification ─────────────────────────────────────
print("\nSample ACTION items (verify ALL 15):")
actions = df[df["action_item"] == 1]
sample_size = min(15, len(actions))
for t in actions["text"].sample(sample_size, random_state=42):
    print(f"  ACTION: {t[:100]}")

print("\nSample NON-ACTION items (check for false negatives):")
non = df[df["action_item"] == 0]
for t in non["text"].sample(10, random_state=42):
    print(f"  OTHER:  {t[:100]}")

# ── Rate assessment ───────────────────────────────────────────
print("\n" + "="*65)
print(f"Positive rate: {rate:.1f}%")
print(f"AMI rate was : ~18%")
if 10 <= rate <= 25:
    print("GOOD — rate is in acceptable range.")
    print("If action samples above look correct: run split_icsi.py")
elif rate < 10:
    print("Still too low — some patterns are still too strict.")
    print("Tell me which NON-ACTION samples look like action items")
    print("and I will add them to positive patterns.")
elif rate > 25:
    print("Still too high — some patterns are too broad.")
    print("Tell me which ACTION samples look WRONG and I will fix.")

# ── Save ──────────────────────────────────────────────────────
df.to_csv("data/processed/icsi_labeled.csv", index=False)
df.to_csv("data/processed/meetingbank_labeled.csv", index=False)
print(f"\nSaved: data/processed/icsi_labeled.csv")
print(f"Next : python src/nlp/split_icsi.py")