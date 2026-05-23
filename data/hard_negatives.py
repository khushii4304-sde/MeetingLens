# hard_negatives.py
# Generates ~1000 scalable hard negatives automatically

import pandas as pd
import random

random.seed(42)

subjects = [
    "I", "We", "The team", "Someone", "They",
    "Engineering", "Management", "The client"
]

modal_phrases = [
    "will probably",
    "should maybe",
    "might",
    "could potentially",
    "would likely",
    "may eventually",
    "can possibly",
    "seems to",
]

activities = [
    "revisit the architecture",
    "look into the issue",
    "improve the workflow",
    "change the process",
    "review the proposal",
    "update the documentation",
    "discuss the timeline",
    "consider redesigning the module",
    "analyze the metrics",
    "investigate the failure",
    "optimize the queries",
    "refactor the backend",
    "rework the dashboard",
    "expand the feature set",
    "prepare the quarterly report",
    "review the deployment logs",
    "update the testing pipeline",
    "evaluate the client feedback",
    "analyze the security concerns",
    "schedule another discussion",
    "document the API endpoints",
    "improve the onboarding process",
    "revisit the sprint planning",
    "check the authentication flow",
    "monitor the production metrics",
    "clean up the old code",
    "review the analytics dashboard",
    "fix the integration issues",
    "improve system reliability",
    "optimize resource allocation",
    "validate the configuration",
    "investigate memory usage",
    "improve meeting coordination",
    "assess the deployment strategy",
    "review stakeholder concerns",
    "evaluate user engagement",
    "restructure the backend logic",
    "discuss scaling limitations",
    "review architectural decisions",
    "update monitoring systems",
    "analyze latency issues",
    "investigate server failures",
    "improve cross-team communication", 
    "evaluate long-term maintainability",
]

endings = [
    "if needed",
    "at some point",
    "when possible",
    "depending on the situation",
    "if things go well",
    "later on",
    "eventually",
    "in the future",
    "after more discussion",
    "if we have time",
    "before the next release",
    "after stakeholder approval",
    "once priorities are clarified",
    "if resources become available",
    "depending on customer feedback",
    "after additional review",
    "once testing is complete",
    "if the roadmap changes",
    "when the team has bandwidth",
    "after the migration finishes",
    "before final deployment",
    "if management approves",
]

past_templates = [
    "We already {}.",
    "Someone should have {} earlier.",
    "The team was supposed to {} last sprint.",
    "I was going to {} yesterday.",
]

question_templates = [
    "Should we {}?",
    "Can we really {}?",
    "Would it make sense to {}?",
    "Will this actually {}?",
]

observation_templates = [
    "The current system seems inefficient.",
    "This approach could be problematic.",
    "There might be issues with the deployment.",
    "The workflow feels unnecessarily complex.",
    "Performance would probably improve with optimization.",
]

hard_negatives = []

# Generate vague future / hypothetical negatives
for _ in range(1400):
    s = random.choice(subjects)
    m = random.choice(modal_phrases)
    a = random.choice(activities)
    e = random.choice(endings)

    sentence = f"{s} {m} {a} {e}."
    hard_negatives.append(sentence)

# Generate past-tense misleading negatives
for _ in range(300):
    template = random.choice(past_templates)
    activity = random.choice(activities)

    hard_negatives.append(template.format(activity))

# Generate question-style negatives
for _ in range(250):
    template = random.choice(question_templates)
    activity = random.choice(activities)

    hard_negatives.append(template.format(activity))

# Add observations
hard_negatives.extend(observation_templates * 10)

# Remove duplicates
hard_negatives = list(set(hard_negatives))

print(f"Generated hard negatives: {len(hard_negatives)}")

# Load original training data
train_df = pd.read_csv("data/processed/train_labeled.csv")

# Create dataframe
hard_neg_df = pd.DataFrame({
    "text": hard_negatives,
    "action_item": 0,
    "meeting_id": "HARD_NEG_SYNTH"
})

# Add missing columns
for col in train_df.columns:
    if col not in hard_neg_df.columns:
        hard_neg_df[col] = None

# Merge
combined = pd.concat([train_df, hard_neg_df], ignore_index=True)

# Shuffle
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
combined.to_csv(
    "data/processed/train_labeled_hardneg.csv",
    index=False
)

print(f"Original dataset : {len(train_df)}")
print(f"Hard negatives   : {len(hard_neg_df)}")
print(f"Final dataset    : {len(combined)}")

print("\nSaved to:")
print("data/processed/train_labeled_hardneg.csv")