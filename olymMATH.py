import pandas as pd

df = pd.read_json("hf://datasets/RUC-AIBOX/OlymMATH/data/OlymMATH-EN-EASY.jsonl", lines=True)

print(df.head())

df.head(10).to_json("olymMATH_10.json", orient="records", indent=2)