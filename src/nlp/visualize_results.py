# src/nlp/visualize_results.py
# Generates 10 research figures for the MeetingLens paper
# Saves all figures to data/processed/results/figures/

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("data/processed/results/figures", exist_ok=True)

# ── Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

COLORS = {
    "ami":        "#2196F3",
    "ood":        "#FF5722",
    "gap":        "#9C27B0",
    "tfidf":      "#607D8B",
    "bert":       "#FF9800",
    "roberta":    "#4CAF50",
    "distilbert": "#00BCD4",
    "positive":   "#4CAF50",
    "neutral":    "#9E9E9E",
    "negative":   "#F44336",
    "decision":   "#3F51B5",
    "action":     "#E91E63",
}

SAVE = "data/processed/results/figures"

# ── Load results ───────────────────────────────────────────────
def load(path):
    with open(path) as f:
        return json.load(f)

def load_model(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

ablation      = load("data/processed/results/ablation_results.json")
multitask     = load("data/processed/results/multitask_results.json")
zeroshot      = load("data/processed/results/zeroshot_results.json")
roberta_res   = load_model("models/roberta_base/test_results.json")
distilbert_res= load_model("models/distilbert_base/test_results.json")

print("Loaded all result files.")
print("Generating figures...\n")

# ── Helper ─────────────────────────────────────────────────────
def get_val(d, *keys, default=0, multiply=1):
    for k in keys:
        if k in d:
            v = d[k]
            return v * multiply if v <= 1 else v
    return default

# =============================================================
# FIGURE 1 — Main results: all 6 model conditions
# =============================================================
fig, ax = plt.subplots(figsize=(13, 6))

models = [
    "TF-IDF\n(AMI only)",
    "TF-IDF\n(AMI+OOD)",
    "BERT\n(AMI only)",
    "BERT\n(AMI+OOD)",
    "RoBERTa\n(AMI+OOD)",
    "DistilBERT\n(AMI+OOD)",
]
ami_scores = [95.3, 81.5, 99.5, 98.0, 96.7, 98.1]
ood_scores = [42.4, 74.9, 44.7, 86.8, 86.5, 86.6]
gaps       = [52.9,  6.6, 54.7, 11.3, 10.1, 11.5]

x     = np.arange(len(models))
width = 0.3

bars_ami = ax.bar(x - width/2, ami_scores, width,
                  label="AMI F1 (%)",  color=COLORS["ami"],
                  alpha=0.85, zorder=3)
bars_ood = ax.bar(x + width/2, ood_scores, width,
                  label="OOD F1 (%)", color=COLORS["ood"],
                  alpha=0.85, zorder=3)

for bar, gap in zip(bars_ood, gaps):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1.5,
            f"gap\n{gap}%",
            ha="center", va="bottom",
            fontsize=7.5, color=COLORS["gap"], fontweight="bold")

for bar in bars_ami:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() - 3,
            f"{bar.get_height():.1f}",
            ha="center", va="top",
            fontsize=8, color="white", fontweight="bold")

for bar in bars_ood:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2,
            max(h - 3, 3),
            f"{h:.1f}",
            ha="center", va="top",
            fontsize=8, color="white", fontweight="bold")

ax.axvline(x=0.5, color="#BDBDBD", linewidth=1, linestyle="--", zorder=1)
ax.axvline(x=2.5, color="#BDBDBD", linewidth=1, linestyle="--", zorder=1)
ax.axhline(y=86.5, color=COLORS["roberta"], linewidth=1,
           linestyle=":", alpha=0.5)
ax.text(5.45, 87.8, "~86.5% OOD ceiling",
        fontsize=8, color=COLORS["roberta"], ha="right")

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel("Macro-F1 (%)")
ax.set_title("Figure 1: AMI vs Out-of-Domain F1 — All Models and Training Conditions")
ax.set_ylim(0, 112)
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3, zorder=0)

plt.tight_layout()
plt.savefig(f"{SAVE}/fig1_main_results.png", bbox_inches="tight")
plt.close()
print("Saved: fig1_main_results.png")

# =============================================================
# FIGURE 2 — Generalization gap horizontal bars
# =============================================================
fig, ax = plt.subplots(figsize=(10, 5))

gap_models = [
    "TF-IDF (AMI only)",
    "BERT (AMI only)",
    "TF-IDF (AMI+OOD)",
    "BERT (AMI+OOD)",
    "DistilBERT (AMI+OOD)",
    "RoBERTa (AMI+OOD)",
]
gap_values = [52.9, 54.7, 6.6, 11.3, 11.5, 10.1]
bar_colors = [
    COLORS["tfidf"], COLORS["bert"],
    COLORS["tfidf"], COLORS["bert"],
    COLORS["distilbert"], COLORS["roberta"],
]

bars = ax.barh(gap_models, gap_values,
               color=bar_colors, alpha=0.85, zorder=3)

for bar, val in zip(bars, gap_values):
    ax.text(bar.get_width() + 0.5,
            bar.get_y() + bar.get_height()/2,
            f"{val}%", va="center",
            fontsize=10, fontweight="bold")

ax.axvline(x=15, color="#F44336", linewidth=1.5,
           linestyle="--", label="Target: gap < 15%")
ax.axhline(y=1.5, color="#BDBDBD", linewidth=1, linestyle=":")

ax.text(62, 0.7, "AMI only",   fontsize=8,
        color="#9E9E9E", ha="right", style="italic")
ax.text(62, 3.8, "AMI + OOD", fontsize=8,
        color="#9E9E9E", ha="right", style="italic")

ax.set_xlabel("Generalization Gap (AMI F1 - OOD F1) — Lower is Better")
ax.set_title("Figure 2: Generalization Gap by Model and Training Data")
ax.set_xlim(0, 65)
ax.legend()
ax.grid(axis="x", alpha=0.3, zorder=0)

plt.tight_layout()
plt.savefig(f"{SAVE}/fig2_generalization_gap.png", bbox_inches="tight")
plt.close()
print("Saved: fig2_generalization_gap.png")

# =============================================================
# FIGURE 3 — Ablation: training data composition
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

a1 = ablation["ablation_1_data_composition"]
categories = ["AMI F1", "OOD F1", "Gap"]

tfidf_ami_only = [a1["tfidf_ami_only_ami_f1"],
                  a1["tfidf_ami_only_ood_f1"],
                  a1["tfidf_ami_only_gap"]]
tfidf_merged   = [a1["tfidf_merged_ami_f1"],
                  a1["tfidf_merged_ood_f1"],
                  a1["tfidf_merged_gap"]]

bert_ami_only  = [a1["bert_ami_only_ami_f1"],
                  a1["bert_ami_only_ood_f1"],
                  a1["bert_ami_only_gap"]]
bert_merged    = [a1["bert_merged_ami_f1"],
                  a1["bert_merged_ood_f1"],
                  a1["bert_merged_gap"]]

x = np.arange(3)
w = 0.35

for ax_sub, ami_only, merged, title, c1, c2 in [
    (axes[0], tfidf_ami_only, tfidf_merged,
     "TF-IDF: AMI only vs AMI+OOD",
     "#607D8B", "#4CAF50"),
    (axes[1], bert_ami_only,  bert_merged,
     "BERT: AMI only vs AMI+OOD",
     "#FF9800", "#4CAF50"),
]:
    ax_sub.bar(x - w/2, ami_only, w,
               label="AMI only", color=c1, alpha=0.85)
    ax_sub.bar(x + w/2, merged,   w,
               label="AMI + OOD", color=c2, alpha=0.85)

    for i, (v1, v2) in enumerate(zip(ami_only, merged)):
        ax_sub.text(i - w/2, v1 + 1, f"{v1}",
                    ha="center", fontsize=9, fontweight="bold")
        ax_sub.text(i + w/2, v2 + 1, f"{v2}",
                    ha="center", fontsize=9, fontweight="bold")

    ax_sub.set_xticks(x)
    ax_sub.set_xticklabels(categories)
    ax_sub.set_ylabel("Score (%)")
    ax_sub.set_title(title)
    ax_sub.set_ylim(0, 112)
    ax_sub.legend()
    ax_sub.grid(axis="y", alpha=0.3)

fig.suptitle("Figure 3: Ablation — Effect of Cross-Corpus Training Data",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{SAVE}/fig3_ablation_data.png", bbox_inches="tight")
plt.close()
print("Saved: fig3_ablation_data.png")

# =============================================================
# FIGURE 4 — Architecture comparison (same data)
# =============================================================
fig, ax = plt.subplots(figsize=(9, 5))

a2 = ablation["ablation_2_architecture"]
arch_models = ["TF-IDF+LR", "BERT-base", "DistilBERT-base", "RoBERTa-base"]
arch_ood    = [a2["tfidf_merged_ood_f1"],
               a2["bert_merged_ood_f1"],
               a2["distilbert_ood_f1"],
               a2["roberta_ood_f1"]]
arch_params = ["sparse", "110M", "66M", "125M"]
arch_colors = [COLORS["tfidf"], COLORS["bert"],
               COLORS["distilbert"], COLORS["roberta"]]

bars = ax.bar(arch_models, arch_ood,
              color=arch_colors, alpha=0.85,
              zorder=3, width=0.5)

for bar, val, params in zip(bars, arch_ood, arch_params):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{val}%", ha="center", va="bottom",
            fontsize=11, fontweight="bold")
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() - 4,
            params, ha="center", va="top",
            fontsize=8, color="white")

ax.set_ylabel("OOD Macro-F1 (%)")
ax.set_title("Figure 4: OOD Performance by Architecture\n"
             "(All trained on AMI + MeetingBank merged data)")
ax.set_ylim(60, 95)
ax.grid(axis="y", alpha=0.3, zorder=0)

ax.annotate("Transformer models\nconverge to ~86-87%",
            xy=(1, 86.8), xytext=(2.5, 81),
            arrowprops=dict(arrowstyle="->", color="#555"),
            fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#FFF9C4", alpha=0.8))

plt.tight_layout()
plt.savefig(f"{SAVE}/fig4_architecture_comparison.png",
            bbox_inches="tight")
plt.close()
print("Saved: fig4_architecture_comparison.png")

# =============================================================
# FIGURE 5 — Efficiency vs performance scatter
# =============================================================
fig, ax = plt.subplots(figsize=(8, 6))

scatter_data = [
    ("TF-IDF+LR",    0,   74.9, 6.6,  COLORS["tfidf"]),
    ("BERT-base",    110, 86.8, 11.3, COLORS["bert"]),
    ("DistilBERT",   66,  86.6, 11.5, COLORS["distilbert"]),
    ("RoBERTa",      125, 86.5, 10.1, COLORS["roberta"]),
]

for name, params, ood_f1, gap, color in scatter_data:
    ax.scatter(params, ood_f1, s=350, color=color,
               zorder=5, alpha=0.9)
    ax.annotate(f"{name}\ngap={gap}%",
                xy=(params, ood_f1),
                xytext=(params + 4, ood_f1 - 1.8),
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor=color, alpha=0.15))

ax.set_xlabel("Model Parameters (Millions) — Lower = More Efficient")
ax.set_ylabel("OOD Macro-F1 (%) — Higher = Better")
ax.set_title("Figure 5: Efficiency vs Generalization Tradeoff\n"
             "(All models trained on AMI + MeetingBank)")
ax.set_xlim(-10, 145)
ax.set_ylim(65, 92)
ax.axhline(y=86, color="#4CAF50", linewidth=1,
           linestyle="--", alpha=0.5)
ax.text(130, 86.4, "86% threshold",
        fontsize=8, color="#4CAF50", ha="right")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{SAVE}/fig5_efficiency_tradeoff.png",
            bbox_inches="tight")
plt.close()
print("Saved: fig5_efficiency_tradeoff.png")

# =============================================================
# FIGURE 6 — Zero-shot gap
# =============================================================
fig, ax = plt.subplots(figsize=(8, 5))

tfidf_ami_p = get_val(zeroshot, "tfidf_ami_f1",
                      "tfidf_ami_only_ami_f1", default=95.3, multiply=100)
tfidf_ood_p = get_val(zeroshot, "tfidf_ood_f1",
                      "tfidf_ami_only_ood_f1", default=42.4, multiply=100)
bert_ami_p  = get_val(zeroshot, "bert_ami_f1",
                      "bert_ami_only_ami_f1",  default=99.5, multiply=100)
bert_ood_p  = get_val(zeroshot, "bert_ood_f1",
                      "bert_ami_only_ood_f1",  default=44.7, multiply=100)

zs_models = ["TF-IDF + LR", "BERT-base"]
zs_ami    = [tfidf_ami_p, bert_ami_p]
zs_ood    = [tfidf_ood_p, bert_ood_p]

x = np.arange(2)
w = 0.3

b1 = ax.bar(x - w/2, zs_ami, w, label="AMI F1 (in-domain)",
            color=COLORS["ami"], alpha=0.85)
b2 = ax.bar(x + w/2, zs_ood, w, label="OOD F1 (zero-shot)",
            color=COLORS["ood"], alpha=0.85)

for i, (ami, ood) in enumerate(zip(zs_ami, zs_ood)):
    gap = ami - ood
    ax.annotate("",
                xy=(i + w/2, ood),
                xytext=(i + w/2, ami),
                arrowprops=dict(arrowstyle="<->",
                                color=COLORS["gap"], lw=2))
    ax.text(i + w/2 + 0.06, (ami + ood)/2,
            f"Gap\n{gap:.1f}%",
            color=COLORS["gap"], fontsize=9,
            fontweight="bold", va="center")

for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f"{bar.get_height():.1f}%",
            ha="center", fontsize=9, fontweight="bold")
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f"{bar.get_height():.1f}%",
            ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(zs_models)
ax.set_ylabel("Macro-F1 (%)")
ax.set_title("Figure 6: Zero-Shot Generalization Gap\n"
             "(AMI-trained models tested on MeetingBank without retraining)")
ax.set_ylim(0, 115)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{SAVE}/fig6_zeroshot_gap.png", bbox_inches="tight")
plt.close()
print("Saved: fig6_zeroshot_gap.png")

# =============================================================
# FIGURE 7 — Sentiment analysis
# =============================================================
mt   = multitask
sent = mt["sentiment_analysis"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sent_labels = ["Positive", "Neutral", "Negative"]
sent_sizes  = [sent["positive_%"],
               sent["neutral_%"],
               sent["negative_%"]]
sent_colors = [COLORS["positive"],
               COLORS["neutral"],
               COLORS["negative"]]

wedges, texts, autotexts = axes[0].pie(
    sent_sizes, labels=sent_labels,
    colors=sent_colors, autopct="%1.1f%%",
    startangle=90, pctdistance=0.75,
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight("bold")

axes[0].set_title("Utterance Sentiment Distribution (AMI Corpus)")

ai_by_sent = sent["action_item_rate_by_sentiment"]
sent_keys  = list(ai_by_sent.keys())
sent_vals  = list(ai_by_sent.values())
bar_cols   = [COLORS.get(k, "#9E9E9E") for k in sent_keys]

bars = axes[1].bar(sent_keys, sent_vals,
                   color=bar_cols, alpha=0.85, width=0.4)
for bar, val in zip(bars, sent_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{val:.1f}%",
                 ha="center", fontsize=10, fontweight="bold")

axes[1].set_ylabel("Action Item Rate (%)")
axes[1].set_title("Action Item Rate by Utterance Sentiment")
axes[1].set_ylim(0, max(sent_vals) * 1.3)
axes[1].grid(axis="y", alpha=0.3)

fig.suptitle("Figure 7: Multi-Task Analysis — Sentiment",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{SAVE}/fig7_sentiment_analysis.png",
            bbox_inches="tight")
plt.close()
print("Saved: fig7_sentiment_analysis.png")

# =============================================================
# FIGURE 8 — Decision detection
# =============================================================
dd = mt["decision_detection"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

total = dd["total_utterances"]
decs  = dd["decisions_detected"]
acts  = int(total * 0.225)
both  = dd["co_occurrence_with_actions"]
other = max(total - decs - acts + both, 0)

sizes  = [acts, decs, both, other]
labels = [
    f"Action Items Only\n({acts/total*100:.1f}%)",
    f"Decisions Only\n({decs/total*100:.1f}%)",
    f"Both\n({both/total*100:.1f}%)",
    f"Neither\n({other/total*100:.1f}%)",
]
pie_colors = [COLORS["action"], COLORS["decision"],
              "#9C27B0", "#BDBDBD"]

valid = [(s, l, c) for s, l, c in
         zip(sizes, labels, pie_colors) if s > 0]
s_v, l_v, c_v = zip(*valid)

axes[0].pie(s_v, labels=l_v, colors=c_v,
            autopct="%1.1f%%", startangle=90,
            pctdistance=0.75)
axes[0].set_title("Utterance Classification Distribution")

bar_cats   = ["Total", "Action Items", "Decisions", "Both"]
bar_vals   = [total, acts, decs, both]
bar_cols_d = ["#78909C", COLORS["action"],
              COLORS["decision"], "#9C27B0"]

bars = axes[1].bar(bar_cats, bar_vals,
                   color=bar_cols_d, alpha=0.85, width=0.5)
for bar, val in zip(bars, bar_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + total * 0.005,
                 f"{int(val):,}",
                 ha="center", fontsize=9, fontweight="bold")

axes[1].set_ylabel("Count")
axes[1].set_title("Multi-Task Label Counts")
axes[1].grid(axis="y", alpha=0.3)

fig.suptitle("Figure 8: Multi-Task Analysis — Decision Detection",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{SAVE}/fig8_decision_detection.png",
            bbox_inches="tight")
plt.close()
print("Saved: fig8_decision_detection.png")

# =============================================================
# FIGURE 9 — Training curves (RoBERTa + DistilBERT)
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, res, name, color in [
    (axes[0], roberta_res,    "RoBERTa-base",    COLORS["roberta"]),
    (axes[1], distilbert_res, "DistilBERT-base", COLORS["distilbert"]),
]:
    if res and "training_history" in res:
        history = res["training_history"]
        epochs  = [h["epoch"]      for h in history]
        losses  = [h["train_loss"] for h in history]
        val_f1s = [h["val_macro_f1"] * 100 for h in history]

        ax2 = ax.twinx()

        ax.plot(epochs, losses, color=color, linewidth=2.5,
                marker="o", markersize=6, label="Train Loss")
        ax2.plot(epochs, val_f1s, color="#F44336", linewidth=2.5,
                 marker="s", markersize=6, linestyle="--",
                 label="Val Macro-F1")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss", color=color)
        ax2.set_ylabel("Val Macro-F1 (%)", color="#F44336")
        ax.set_title(f"{name} Training Curve")
        ax.grid(alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  loc="center right", fontsize=8)

        ax.annotate(f"Final loss: {losses[-1]:.4f}",
                    xy=(epochs[-1], losses[-1]),
                    xytext=(epochs[-1] - 0.3, losses[-1] + 0.02),
                    fontsize=8)
        ax2.annotate(f"Final F1: {val_f1s[-1]:.1f}%",
                     xy=(epochs[-1], val_f1s[-1]),
                     xytext=(epochs[-1] - 0.3, val_f1s[-1] - 3),
                     fontsize=8, color="#F44336")
    else:
        ax.text(0.5, 0.5, f"{name}\nhistory not available",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=11)

fig.suptitle("Figure 9: Training Loss and Validation F1 Curves",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{SAVE}/fig9_training_curves.png",
            bbox_inches="tight")
plt.close()
print("Saved: fig9_training_curves.png")

# =============================================================
# FIGURE 10 — Gap reduction story (key narrative)
# =============================================================
fig, ax = plt.subplots(figsize=(11, 6))

stages = [
    "TF-IDF\n(AMI only)",
    "BERT\n(AMI only)",
    "BERT\n(AMI+OOD)",
    "DistilBERT\n(AMI+OOD)",
    "RoBERTa\n(AMI+OOD)",
]
gap_prog = [52.9, 54.7, 11.3, 11.5, 10.1]
ood_prog = [42.4, 44.7, 86.8, 86.6, 86.5]

x = np.arange(len(stages))

ax.plot(x, gap_prog, color=COLORS["gap"], linewidth=2.5,
        marker="o", markersize=8,
        label="Generalization Gap (%)", zorder=5)
ax.plot(x, ood_prog, color=COLORS["ood"], linewidth=2.5,
        marker="s", markersize=8, linestyle="--",
        label="OOD F1 (%)", zorder=5)

ax.fill_between(x, gap_prog, alpha=0.08, color=COLORS["gap"])
ax.fill_between(x, ood_prog, alpha=0.08, color=COLORS["ood"])

for i, (gap, ood) in enumerate(zip(gap_prog, ood_prog)):
    ax.text(i, gap + 1.8, f"{gap}%",
            ha="center", fontsize=9,
            color=COLORS["gap"], fontweight="bold")
    ax.text(i, ood + 1.8, f"{ood}%",
            ha="center", fontsize=9,
            color=COLORS["ood"], fontweight="bold")

ax.annotate("Cross-corpus\ntraining added",
            xy=(2, 11.3), xytext=(1.1, 32),
            arrowprops=dict(arrowstyle="->", color="#555"),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#FFF9C4", alpha=0.9))

ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=9)
ax.set_ylabel("Score (%)")
ax.set_title(
    "Figure 10: Research Story — Gap Reduction Through Cross-Corpus Training\n"
    "Gap shrinks from ~54% to ~10%  |  OOD F1 rises from ~43% to ~87%"
)
ax.legend(loc="center right")
ax.grid(alpha=0.3)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(f"{SAVE}/fig10_gap_reduction_story.png",
            bbox_inches="tight")
plt.close()
print("Saved: fig10_gap_reduction_story.png")

# ── Summary ────────────────────────────────────────────────────
print()
print("=" * 55)
print("All 10 figures generated successfully")
print("=" * 55)
print(f"Location: {SAVE}/")
print()
figures = sorted([f for f in os.listdir(SAVE) if f.endswith(".png")])
for f in figures:
    print(f"  {f}")
print()
print("Figure guide for your paper:")
print("  fig1  — Main results (all 6 model conditions)")
print("  fig2  — Gap comparison horizontal bars")
print("  fig3  — Ablation: data composition")
print("  fig4  — Architecture comparison")
print("  fig5  — Efficiency vs generalization scatter")
print("  fig6  — Zero-shot gap")
print("  fig7  — Sentiment analysis")
print("  fig8  — Decision detection")
print("  fig9  — Training curves")
print("  fig10 — Gap reduction story (key narrative figure)")