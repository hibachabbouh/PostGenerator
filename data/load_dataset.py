
from datasets import load_dataset
import pandas as pd
import os
dataset = load_dataset("Waterfront/social-media-captions")
df = pd.DataFrame(dataset["train"])
df_sampled = df.sample(n=100, random_state=42)
os.makedirs("data/raw", exist_ok=True)
df_sampled.to_csv("data/raw/social_media_captions.csv", index=False)
print(" Dataset saved to data/raw/social_media_captions.csv")


