import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import math
from yummygpt import TransformerFinal
import tiktoken
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("-path", "--path", type=str, required=True)

args = parser.parse_args()

model_saved = torch.load(f"saved_models/{args.path}.pth", map_location=torch.device(device))

torch.manual_seed(model_saved["hyper_params"]["torch_seed"])

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
eot = tokenizer.eot_token

if __name__ == "__main__":
    m = TransformerFinal(model_saved["hyper_params"]["d_model"], model_saved["hyper_params"]["context_len"], model_saved["hyper_params"]["n_heads"], model_saved["hyper_params"]["n_blocks"], model_saved["hyper_params"]["vocab_size"], model_saved["hyper_params"]["dropout"], model_saved["hyper_params"]["weight_tying"]).to(device)
    print("loading state dicts....")
    m.load_state_dict(model_saved["model_state_dict"])
    print("done")
    
    while True:
        in_text = input("\n->:")
        if in_text == "exit":
            break

        tokens = torch.tensor(tokenizer.encode(in_text, allowed_special="all")).unsqueeze(0)

        for i in range(200):
            out_probs = m.generate_tokens(tokens, context_len=model_saved["hyper_params"]["context_len"], temp=0.7)
            out_token = torch.multinomial(out_probs, num_samples=1)  # sample from the probabilities
            tokens = torch.cat((tokens, out_token), dim=1)

            print(tokenizer.decode(out_token[0].tolist()), flush=True, end="")

            #if i % 50 == 0:
            #    print(out_probs[:, 50256])

            if out_token.item() == eot:
                break
