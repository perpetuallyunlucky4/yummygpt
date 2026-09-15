from datasets import load_dataset
import tiktoken
import numpy as np

tokenizer_base = tiktoken.get_encoding("gpt2")
special_toks = {
    **tokenizer_base._special_tokens,
    "<|user|>": 50257,
    "<|assistant|>": 50258,
    "<|system|>": 50259,
}
tokenizer = tiktoken.Encoding(
    name="gpt2_chat",
    pat_str=tokenizer_base._pat_str,
    mergeable_ranks=tokenizer_base._mergeable_ranks,
    special_tokens = special_toks,
) #to create a tokenizer with all the special role tokens

dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
dataset = dataset.take(2000000)

print("writing tokens to binary file....")

n_docs = 0

buffer = []

with open("model_train_files/fineweb_tokens.bin", "wb") as outfile:
    for document in dataset:
        text = document["text"] + "<|endoftext|>\n"
        tokens = tokenizer.encode(text, allowed_special="all")
        buffer.extend(tokens)
        if len(buffer) >= 1_00_000:
            np.array(buffer, dtype=np.uint16).tofile(outfile)
            buffer.clear()
        n_docs += 1
        if n_docs % 10000 == 0:
            print(f"{n_docs} documents processed")

    if buffer:
        np.array(buffer, dtype=np.uint16).tofile(outfile)
print("done")
