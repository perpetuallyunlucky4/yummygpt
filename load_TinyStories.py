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

print("writing data to txt file....")

dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
dataset = dataset.take(20000)

tezt = "<|endoftext|>\n\n".join(dataset["text"]) + "<|endoftext|>"

with open("model_train_files/TinyStories.txt", "w", encoding="utf-8") as f:
    f.write(tezt)

print("done")
print("writing tokens to binary file....")

buffer = ""
doc_count = 0
seperator = "<|endoftext|>\n\n"

with open("model_train_files/TinyStories.txt", "r") as infile, open("model_train_files/TinyStories_tokens.bin", "wb") as outfile:
    while True:
        chunk = infile.read(10_000_000)
        if not chunk:
            break

        buffer += chunk
        docs = buffer.split(seperator)
        buffer = docs.pop()

        for doc in docs:
            text = doc + seperator
            tokens = tokenizer.encode(text, allowed_special="all")
            np.array(tokens, dtype=np.uint16).tofile(outfile)
            doc_count +=1

        print(f"processed {doc_count} documents\r")

    if buffer:
        tokens = tokenizer.encode(text, allowed_special="all")
        np.array(tokens, dtype=np.uint16).tofile(outfile)

print("done")
