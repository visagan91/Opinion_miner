import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import ast

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def run_eda(df):
    print("\U0001F4CA Dataset shape:", df.shape)
    print("\n🔍 Sample rows:")
    print(df[["sentence", "clean_sentence", "feature", "clean_feature", "sentiment", "strength"]].head())

    print("\n🎯 Sentiment distribution:")
    print(df["sentiment"].value_counts())

    print("\n💥 Sentiment strength distribution:")
    print(df["strength"].value_counts())

    print("\n🏷️ Number of unique features:", df["clean_feature"].nunique())
    print("🏷️ Number of unique aspects:", df["aspect"].nunique())
    print("🏷️ Number of unique domains:", df["domain"].nunique() if "domain" in df.columns else "N/A")

    os.makedirs("figures", exist_ok=True)

    # Sentiment distribution
    sns.countplot(data=df, x="sentiment", palette="Set2")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("figures/sentiment_distribution.png")
    plt.clf()

    # Sentiment strength distribution
    sns.countplot(data=df, x="strength", hue="sentiment", palette="Set1")
    plt.title("Sentiment Strength by Polarity")
    plt.xlabel("Strength (1–3)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("figures/sentiment_strength.png")
    plt.clf()

    # Distribution of sentence length
    df["sentence_length"] = df["clean_sentence"].apply(lambda x: len(x.split()))
    sns.histplot(df["sentence_length"], bins=30, kde=True)
    plt.title("Sentence Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("figures/sentence_length_distribution.png")
    plt.clf()

    # POS tag frequency
    if "pos_tags" in df.columns:
        print("\n\U0001F9E0 POS tag frequency (top 15):")
        pos_flat = [token.split("_")[-1] for row in df["pos_tags"].dropna() for token in row.split()]
        pos_series = pd.Series(pos_flat).value_counts().head(15)
        print(pos_series)

        sns.barplot(x=pos_series.values, y=pos_series.index, palette="magma")
        plt.title("Top 15 POS Tags")
        plt.xlabel("Frequency")
        plt.ylabel("POS Tag")
        plt.tight_layout()
        plt.savefig("figures/pos_tag_distribution.png")
        plt.clf()

    # Noun Phrase Distribution (if available)
    if "noun_phrases" in df.columns:
        print("\n🧠 Top Noun Phrases:")
        all_nps = [np for row in df["noun_phrases"].dropna() for np in row.split(";") if np.strip()]
        np_series = pd.Series(all_nps).value_counts().head(15)
        print(np_series)

        sns.barplot(x=np_series.values, y=np_series.index, palette="viridis")
        plt.title("Top 15 Noun Phrases")
        plt.xlabel("Frequency")
        plt.ylabel("Noun Phrase")
        plt.tight_layout()
        plt.savefig("figures/noun_phrase_distribution.png")
        plt.clf()

    # Aspect distribution across domain
    if "domain" in df.columns:
        aspect_domain = df.groupby(["aspect", "domain"]).size().unstack(fill_value=0)
        print("\n📊 Aspect Mention Count by Domain:")
        print(aspect_domain.head(10))

        aspect_domain.T.plot(kind="barh", stacked=True, colormap="tab20")
        plt.title("Aspect Distribution per Domain")
        plt.xlabel("Count")
        plt.ylabel("Domain")
        plt.tight_layout()
        plt.savefig("figures/aspect_per_domain_distribution.png")
        plt.clf()

    print("\n✅ All EDA visualizations saved to /figures")


if __name__ == "__main__":
    df = pd.read_csv("output/cleaned_reviews.csv")
    df.dropna(subset=["clean_sentence", "clean_feature", "sentiment", "strength"], inplace=True)
    run_eda(df)
