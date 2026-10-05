from datasets import load_dataset
import os

dataset = load_dataset("stanfordnlp/sst2")

df = dataset["train"].to_pandas()
print(df.shape)
print(df.head())

df = df.sample(n=10000, random_state=42)
os.makedirs("data", exist_ok=True)
df.to_csv("data/sst2_sample.csv", index=False)
print(df["label"].value_counts(normalize=True))