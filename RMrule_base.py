# RMrule_base.py
import pandas as pd
import spacy
from nltk.corpus import opinion_lexicon
from nltk.tokenize import word_tokenize
import nltk
import os

nltk.download("opinion_lexicon")
nltk.download("punkt")

nlp = spacy.load("en_core_web_sm")

POSITIVE = set(opinion_lexicon.positive())
NEGATIVE = set(opinion_lexicon.negative())

def find_opinion_word(token):
    # Search for adjectives directly related to the feature token
    for child in token.children:
        if child.pos_ == "ADJ":
            return child.text
    if token.head.pos_ == "ADJ":
        return token.head.text
    return None

def rule_based_relation_mapping(df):
    results = []

    for _, row in df.iterrows():
        sentence = row["clean_sentence"]
        aspect = row["aspect"]
        domain = row["domain"]

        if not isinstance(aspect, str) or not isinstance(sentence, str):
            continue

        doc = nlp(sentence)

        for token in doc:
            if token.text.lower() == aspect:
                opinion = find_opinion_word(token)
                if opinion:
                    sentiment = (
                        "positive" if opinion.lower() in POSITIVE else
                        "negative" if opinion.lower() in NEGATIVE else
                        "neutral"
                    )

                    results.append({
                        "domain": domain,
                        "aspect": aspect,
                        "clean_sentence": sentence,
                        "opinion_word": opinion,
                        "sentiment": sentiment,
                        "method": "rule-based"
                    })
                    break  # Avoid duplicate matches in the same sentence

    return pd.DataFrame(results)


if __name__ == "__main__":
    input_path = "output/cleaned_reviews.csv"
    output_path = "output/relation_mapping_rule_based.csv"

    df = pd.read_csv(input_path)
    df.dropna(subset=["clean_sentence", "aspect", "domain"], inplace=True)

    print("🔍 Performing rule-based relation mapping...")
    result_df = rule_based_relation_mapping(df)

    os.makedirs("output", exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"✅ Rule-based relation mapping complete. Saved to {output_path}")
