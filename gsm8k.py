import pandas as pd

splits = {'train': 'main/train-00000-of-00001.parquet', 'test': 'main/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/openai/gsm8k/" + splits["train"])

print(df.head())

df.head(50).to_json("gsm8k_50.json", orient="records", indent=2)