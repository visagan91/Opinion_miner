import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import pi

# ---------- CONFIGURATION ---------- #
method = "bert"  # Choose: 'bert' or 'rule'

input_paths = {
    "bert": "output/relation_mapping_bert_based.csv",
    "rule": "output/relation_mapping_rule_based.csv"
}

output_file = f"output/opinion_summary_{method}.csv"
plot_dir = f"figures/{method}_based"
os.makedirs("output", exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# ---------- LOAD DATA ---------- #
print(f"📥 Using relation mapping from: {input_paths[method]}")
df = pd.read_csv(input_paths[method])

# Rename predicted_sentiment to sentiment for consistency
if method == "bert":
    df = df.rename(columns={"predicted_sentiment": "sentiment"})

# Filter valid rows
df = df.dropna(subset=["aspect", "sentiment"])

# ---------- AGGREGATE ---------- #
summary = df.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0)
summary["total_mentions"] = summary.sum(axis=1)

# Ensure all sentiment columns exist
for label in ["positive", "neutral", "negative"]:
    if label not in summary.columns:
        summary[label] = 0

# Compute sentiment ratios
for label in ["positive", "neutral", "negative"]:
    summary[f"{label}_ratio"] = summary[label] / summary["total_mentions"]

# Save to CSV
summary.to_csv(output_file)
print(f"✅ Opinion summary saved to: {output_file}")

# ---------- PLOTS ---------- #
sns.set(style="whitegrid")

# Function to annotate bars
def annotate_bars(ax):
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.01, p.get_y() + p.get_height()/2,
                f'{width:.2f}' if width < 1 else int(width),
                va='center')

# Top 10 Positive Aspects
plt.figure(figsize=(9, 6))
top_positive = summary.sort_values("positive_ratio", ascending=False).head(10)
ax = sns.barplot(x="positive_ratio", y=top_positive.index, data=top_positive, palette="Greens_r")
annotate_bars(ax)
plt.title(f"Top 10 Positive Aspects ({method.title()}-based)")
plt.xlabel("Positive Ratio")
plt.ylabel("Aspect")
plt.tight_layout()
plt.savefig(f"{plot_dir}/top_positive_aspects.png")
plt.clf()

# Top 10 Negative Aspects
plt.figure(figsize=(9, 6))
top_negative = summary.sort_values("negative_ratio", ascending=False).head(10)
ax = sns.barplot(x="negative_ratio", y=top_negative.index, data=top_negative, palette="Reds_r")
annotate_bars(ax)
plt.title(f"Top 10 Negative Aspects ({method.title()}-based)")
plt.xlabel("Negative Ratio")
plt.ylabel("Aspect")
plt.tight_layout()
plt.savefig(f"{plot_dir}/top_negative_aspects.png")
plt.clf()

# Top 10 Most Mentioned Aspects
plt.figure(figsize=(9, 6))
top_mentioned = summary.sort_values("total_mentions", ascending=False).head(10)
ax = sns.barplot(x="total_mentions", y=top_mentioned.index, data=top_mentioned, palette="Blues_r")
annotate_bars(ax)
plt.title(f"Top 10 Most Mentioned Aspects ({method.title()}-based)")
plt.xlabel("Total Mentions")
plt.ylabel("Aspect")
plt.tight_layout()
plt.savefig(f"{plot_dir}/top_mentioned_aspects.png")
plt.clf()

# Sentiment Composition (Stacked Bar)
plt.figure(figsize=(10, 7))
comp = top_mentioned[["positive", "neutral", "negative"]]
comp.plot(kind="barh", stacked=True, colormap="viridis")
plt.title(f"Sentiment Composition for Top Aspects ({method.title()}-based)")
plt.xlabel("Mentions")
plt.ylabel("Aspect")
plt.tight_layout()
plt.savefig(f"{plot_dir}/aspect_sentiment_composition.png")
plt.clf()

# Radar Chart for Sentiment Ratios
radar_data = summary.sort_values("total_mentions", ascending=False).head(6)
labels = ["positive_ratio", "neutral_ratio", "negative_ratio"]

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]  # close the loop

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for idx, row in radar_data.iterrows():
    values = [row[label] for label in labels]
    values += values[:1]  # repeat first to close
    ax.plot(angles, values, label=idx)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels([label.replace("_ratio", "").title() for label in labels])
ax.set_yticks([0.25, 0.5, 0.75])
ax.set_yticklabels(["25%", "50%", "75%"])
ax.set_title(f"Sentiment Ratios Radar Chart ({method.title()}-based)")
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))
plt.tight_layout()
plt.savefig(f"{plot_dir}/aspect_sentiment_radar.png")
plt.clf()

# Overall Pie Chart
overall_counts = df["sentiment"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(overall_counts, labels=overall_counts.index, autopct="%.1f%%", colors=["green", "grey", "red"])
plt.title(f"Overall Sentiment Distribution ({method.title()}-based)")
plt.tight_layout()
plt.savefig(f"{plot_dir}/overall_sentiment_pie.png")
plt.clf()

print(f"📊 All enhanced plots saved to `{plot_dir}/`:")
