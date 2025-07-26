# ✅ UPDATED: ingest.py
# ---------------------
# Converts raw .txt to CSV, and adds 'domain' column + separate CSVs per dataset folder

import os
import re
import pandas as pd


def parse_review_file(filepath, domain):
    data = []
    current_sentence = ""
    encodings = ['utf-8', 'ISO-8859-1', 'cp1252']

    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith("[t]") or line.startswith("***"):
                        continue

                    if line.startswith("##"):
                        current_sentence = line[2:].strip()
                    else:
                        annotations = re.findall(r"([\w\s\-&]+?)\[(\+|\-)(\d)\]", line)
                        for feature, polarity, strength in annotations:
                            data.append({
                                "domain": domain,
                                "sentence": current_sentence,
                                "feature": feature.strip(),
                                "sentiment": "positive" if polarity == "+" else "negative",
                                "strength": int(strength)
                            })
            break
        except UnicodeDecodeError:
            continue

    return pd.DataFrame(data)


def parse_all_reviews(root_folder):
    all_data = []
    output_root = "output/per_dataset"
    os.makedirs(output_root, exist_ok=True)

    for folder in os.listdir(root_folder):
        dataset_path = os.path.join(root_folder, folder)
        if not os.path.isdir(dataset_path):
            continue

        print(f"📁 Processing folder: {folder}")
        domain_data = []

        for file in os.listdir(dataset_path):
            if file.endswith(".txt"):
                path = os.path.join(dataset_path, file)
                domain = os.path.splitext(file)[0]
                df = parse_review_file(path, domain)
                domain_data.append(df)

        if domain_data:
            df_combined = pd.concat(domain_data, ignore_index=True)
            df_combined.to_csv(f"{output_root}/{folder}.csv", index=False)
            all_data.append(df_combined)

    return pd.concat(all_data, ignore_index=True)


if __name__ == "__main__":
    df_reviews = parse_all_reviews("data")
    os.makedirs("output", exist_ok=True)
    df_reviews.to_csv("output/combined_reviews.csv", index=False)
    print("✅ All reviews saved to output/combined_reviews.csv")
