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

IGNORE = -100

conv_index = []

current_offset = 0
n_conv = 0

with open("model_train_files/axiom.txt", "r", encoding="utf-8") as f:
    text = f.read()

conversations = text.split("ENDTEXT")
print(f"{len(conversations)} conversations")

with open("model_train_files/axiom_tokens.bin", "wb") as outfile, open("model_train_files/axiom_targets.bin", "wb") as targetfile:
    for conversation in conversations:
        conversation = conversation.strip()

        if not conversation:
            continue

        lines = conversation.splitlines()

        input_tokens = []
        target_tokens = []

        role = None

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if line == "USER":
                role = "user"
                toks = tokenizer.encode("<|user|>\n", allowed_special="all")

                input_tokens.extend(toks)
                target_tokens.extend([IGNORE] * len(toks))

            elif line == "ASSISTANT":
                role = "assistant"
                toks = tokenizer.encode("<|assistant|>\n", allowed_special="all")

                input_tokens.extend(toks)
                target_tokens.extend(toks)

            else:
                toks = tokenizer.encode(line + "\n", allowed_special="all")

                input_tokens.extend(toks)

                if role == "user":
                    target_tokens.extend([IGNORE] * len(toks))
                elif role == "assistant":
                    target_tokens.extend(toks)

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

np.save("model_train_files/axiom_index.npy", np.array(conv_index, dtype=np.int64))

print("done")
