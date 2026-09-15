from datasets import load_dataset
import tiktoken
import numpy as np
import torch

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

eot = tokenizer.encode("<|endoftext|>\n", allowed_special="all")

dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)

IGNORE = -100

conv_index = []

current_offset = 0
n_conv = 0

with open("model_train_files/ultrachat_tokens.bin", "wb") as outfile, open("model_train_files/ultrachat_targets.bin", "wb") as targetfile:
    for sample in dataset:
        input_tokens = []
        target_tokens = []

        for message in sample["messages"]:
            role = message["role"]
            if role == "system":
                continue

            if role == "user":
                text = "<|user|>\n" + message["content"].strip() + "\n"
                toks = tokenizer.encode(text, allowed_special="all")

                input_tokens.extend(toks)
                target_tokens.extend([IGNORE] * len(toks))

            elif role == "assistant":
                text = "<|assistant|>\n" + message["content"].strip() + "\n"
                toks = tokenizer.encode(text, allowed_special="all")

                input_tokens.extend(toks)
                target_tokens.extend(toks)

            else:
                continue

        input_tokens.extend(eot)
        target_tokens.extend(eot)

        conv_index.append((current_offset, len(input_tokens)))

        np.array(input_tokens, dtype=np.uint16).tofile(outfile)
        np.array(target_tokens, dtype=np.int32).tofile(targetfile)

        current_offset += len(input_tokens)

        n_conv += 1
        if n_conv % 1000 == 0:
            print(f"{n_conv} messages processed")
            print(target_tokens)

np.save("model_train_files/index.npy", np.array(conv_index, dtype=np.int64))

print("done")

