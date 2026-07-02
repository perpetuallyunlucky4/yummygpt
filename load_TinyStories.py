from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories")

tezt = "<|endoftext|>\n".join(dataset["train"]["text"][:12000]) + "<|endoftext|>"

with open("TinyStories.txt", "w") as f:
  f.write(tezt)
