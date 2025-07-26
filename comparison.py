# comparison.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

# Set styles
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (6, 4)

# Load relation mapping outputs
rule_df = pd.read_csv("output/relation_mapping_rule_based.csv")
bert_df = pd.read_csv("output/relation_mapping_bert_based.csv")

# Standardize column names
rule_df = rule_df.rename(columns={
    "sentiment": "rule_sentiment",
    "clean_sentence": "sentence"
})

bert_df = bert_df.rename(columns={
    "predicted_sentiment": "bert_sentiment",
    "clean_sentence": "sentence"
})

# Merge on sentence + aspect
merged = pd.merge(rule_df, bert_df, on=["sentence", "aspect"], how="inner")

# Agreement analysis
merged["agreement"] = merged["rule_sentiment"] == merged["bert_sentiment"]

# Stats
total = len(merged)
agreement = merged["agreement"].sum()
agreement_rate = agreement / total if total > 0 else 0

print(f"✅ Compared {total} overlapping sentence-aspect pairs")
print(f"🤝 Agreement count: {agreement}")
print(f"📊 Agreement rate: {agreement_rate:.2%}")

# Save comparison results
os.makedirs("output", exist_ok=True)
merged.to_csv("output/comparison_result.csv", index=False)

# Save figures to a dedicated comparison subfolder
figure_dir = "figures/comparison"
os.makedirs(figure_dir, exist_ok=True)

# Confusion Matrix
cm = confusion_matrix(merged["rule_sentiment"], merged["bert_sentiment"],
                      labels=["positive", "neutral", "negative"])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["positive", "neutral", "negative"],
            yticklabels=["positive", "neutral", "negative"])
plt.xlabel("BERT Sentiment")
plt.ylabel("Rule-based Sentiment")
plt.title("Confusion Matrix: Rule vs BERT")
plt.tight_layout()
plt.savefig(f"{figure_dir}/confusion_matrix.png")
plt.clf()

# Agreement count plot
sns.countplot(data=merged, x="agreement", palette="coolwarm")
plt.title("Agreement Between Rule and BERT")
plt.ylabel("Count")
plt.xlabel("Agreement (True/False)")
plt.tight_layout()
plt.savefig(f"{figure_dir}/agreement_distribution.png")
plt.clf()

print("📊 Visuals saved to:")
print(f"   - {figure_dir}/confusion_matrix.png")
print(f"   - {figure_dir}/agreement_distribution.png")
