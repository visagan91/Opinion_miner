import pandas as pd
import re
import os
import nltk
import spacy
from nltk.corpus import stopwords

# Download stopwords (safe to re-run)
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

STOPWORDS = set(stopwords.words('english'))

# ----------------------------
# Aspect Synonym Dictionary
# ----------------------------
ASPECT_SYNONYMS = {
    "battery": ["battery", "battery life", "battery backup", "batteries"],
    "screen": ["screen", "display", "lcd", "monitor"],
    "sound": ["sound", "audio", "speaker", "volume"],
    "camera": ["camera", "photo", "image", "pictures"],
    "price": ["price", "cost", "value", "worth"],
    "performance": ["performance", "speed", "slow", "fast"],
    "memory": ["memory", "storage", "ram", "space"],
    "connectivity": ["wifi", "bluetooth", "connection", "connectivity"],
    "size": ["size", "weight", "dimension"],
    "design": ["design", "look", "build", "style"]
}

REVERSE_MAP = {word: aspect for aspect, words in ASPECT_SYNONYMS.items() for word in words}

def clean_text(text, remove_stopwords=True):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if remove_stopwords:
        tokens = text.split()
        text = " ".join([t for t in tokens if t not in STOPWORDS])

    return text

def normalize_feature(feature):
    return REVERSE_MAP.get(feature.lower(), feature.lower())

def extract_pos_tags(text):
    doc = nlp(text)
    return " ".join([f"{token.text}_{token.pos_}" for token in doc])

def extract_noun_phrases(text):
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]

def preprocess_dataframe(df, remove_stopwords=True):
    df["clean_sentence"] = df["sentence"].apply(lambda x: clean_text(x, remove_stopwords))
    df["clean_feature"] = df["feature"].apply(lambda x: clean_text(x, remove_stopwords=False))
    df["aspect"] = df["clean_feature"].apply(normalize_feature)

    # NEW: POS tags and noun phrases
    df["pos_tags"] = df["clean_sentence"].apply(extract_pos_tags)
    df["noun_phrases"] = df["clean_sentence"].apply(extract_noun_phrases)

    return df

if __name__ == "__main__":
    df = pd.read_csv("output/combined_reviews.csv")

    print("✅ Preprocessing with stopword removal, aspect normalization, POS tagging and NP extraction...")
    df_clean = preprocess_dataframe(df, remove_stopwords=True)

    df_clean = df_clean.dropna(subset=["clean_sentence", "clean_feature", "aspect"])

    print("\n📄 Sample preview:")
    print(df_clean[["domain", "sentence", "clean_sentence", "feature", "clean_feature", "aspect", "pos_tags", "noun_phrases"]].head())

    os.makedirs("output", exist_ok=True)
    df_clean.to_csv("output/cleaned_reviews.csv", index=False)
    print("\n✅ Cleaned & enriched data saved to output/cleaned_reviews.csv")
