# RMmodel_base.py
import pandas as pd
from transformers import pipeline
import os
from tqdm import tqdm

# Load cleaned dataset
df = pd.read_csv("output/cleaned_reviews.csv")

# Filter out rows with usable data
df = df.dropna(subset=["clean_sentence", "aspect", "domain"])
df = df[df["clean_sentence"].str.len() > 5]

# Load BERT-style classifier (BART)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Sentiment classes to predict
sentiment_labels = ["positive", "neutral", "negative"]

results = []
print("🔍 Predicting sentiment using BERT (BART MNLI)...")

# Iterate with progress bar
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
    sentence = row["clean_sentence"]
    aspect = row["aspect"]
    domain = row["domain"]

    hypothesis_template = f"The sentiment toward the {aspect} is {{}}"

    try:
        prediction = classifier(
            sequences=sentence,
            candidate_labels=sentiment_labels,
            hypothesis_template=hypothesis_template
        )
        predicted_label = prediction["labels"][0]
        confidence = prediction["scores"][0]

        results.append({
            "domain": domain,
            "aspect": aspect,
            "clean_sentence": sentence,
            "predicted_sentiment": predicted_label,
            "confidence": round(confidence, 3),
            "method": "bert-based"
        })

        if idx % 100 == 0:
            print(f"✅ Processed {idx} rows")

    except Exception as e:
        print(f"⚠️ Error on row {idx}: {e}")
        continue

# Save results
os.makedirs("output", exist_ok=True)
out_path = "output/relation_mapping_bert_based.csv"
pd.DataFrame(results).to_csv(out_path, index=False)

print(f"\n✅ BERT-based relation mapping complete.")
print(f"📄 Output saved to: {out_path}")
