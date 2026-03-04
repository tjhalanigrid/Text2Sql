from datasets import load_dataset

ds = load_dataset("xlangai/spider")

print(ds)
print(ds["train"][0])

