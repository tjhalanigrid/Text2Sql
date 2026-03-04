from datasets import Dataset
import pandas as pd

df = pd.read_csv("../data/processed/train.csv")
ds = Dataset.from_pandas(df)
ds.save_to_disk("../data/processed/train")
print("DONE")

