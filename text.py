# quick_check_columns.py
import pandas as pd

rule_path = "output/relation_mapping_rule_based.csv"
bert_path = "output/relation_mapping_bert_based.csv"

rule_df = pd.read_csv(rule_path)
bert_df = pd.read_csv(bert_path)

print("📄 Rule-based Columns:", list(rule_df.columns))
print("📄 BERT-based Columns:", list(bert_df.columns))
