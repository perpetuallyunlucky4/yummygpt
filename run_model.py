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

tokenizer = tiktoken.get_encoding("gpt2")
eot = tokenizer.eot_token

if __name__ == "__main__":
    m = TransformerFinal(model_saved["hyper_params"]["d_model"], model_saved["hyper_params"]["context_len"], model_saved["hyper_params"]["n_heads"], model_saved["hyper_params"]["n_blocks"], model_saved["hyper_params"]["vocab_size"], model_saved["hyper_params"]["dropout"], model_saved["hyper_params"]["weight_tying"]).to(device)
    print("loading state dicts....")
    m.load_state_dict(model_saved["model_state_dict"])
    print("done")
    
    while True:
        in_text = input("->:")
        if in_text == "exit":
            break

        tokens = torch.tensor(tokenizer.encode(in_text, allowed_special={"<|endoftext|>"})).unsqueeze(0)

        with torch.no_grad():
            out_tokens = m.generate_tokens(tokens, max_iters=500, temp=0.7, eos=eot)

        print(out_tokens)
        print(f"->:{tokenizer.decode(out_tokens[0].tolist())}")
