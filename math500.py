import pandas as pd

# Load the dataset
df = pd.read_json("hf://datasets/HuggingFaceH4/MATH-500/test.jsonl", lines=True)

# Filter for levels 1, 2, and 3
df_filtered = df[df['level'].isin([1, 2, 3])]

print(f"Original dataset size: {len(df)}")
print(f"Filtered dataset size: {len(df_filtered)}")
print("\nSample of filtered data:")
print(df_filtered.head())

# Save the filtered dataset
df_filtered.to_json("MATH-500_123.json", orient="records", indent=2)