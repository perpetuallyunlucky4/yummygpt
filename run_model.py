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
parser.add_argument("-m", "--mode", default="text")
parser.add_argument("-s", "--samples", type=int, default=1)
parser.add_argument("-o", "--outfile", default=None)
parser.add_argument("-t", "--temperature", type=float, default=1.0)
parser.add_argument("-p", "--topp", type=float)
parser.add_argument("-r", "--reppenalty", type=float, default=1.0)
parser.add_argument("-max", "--maxtokens", type=int, default=500)

args = parser.parse_args()
samples = args.samples
mode = args.mode
outfile = args.outfile
temp = args.temperature
topp = args.topp
rep_penalty = args.reppenalty
max_tokens = args.maxtokens

model_saved = torch.load(f"saved_models/{args.path}", map_location=torch.device(device))

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
    m.eval()

    final_tokens = torch.empty((1, 0), dtype=torch.long, device=device)

    while True:
        in_text = input("\n->:")
        if in_text == "exit":
            break

        if mode == "instruct" or mode == "greedy-instruct":
            in_text = "<|user|>\n" + in_text + "\n<|assistant|>\n"

        for sample in range(samples):
            tokens = torch.tensor(tokenizer.encode(in_text, allowed_special="all")).unsqueeze(0).to(device)
            final_tokens = torch.cat((final_tokens, tokens), dim=-1)
            for i in range(max_tokens):
                out_probs = m.generate_tokens(final_tokens, context_len=model_saved["hyper_params"]["context_len"], temp=temp, rep_penalty=rep_penalty)

                if topp is not None:
                    sorted_probs, sorted_indexes = torch.sort(out_probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    mask = cumulative_probs > topp
                    mask[:, 1:] = mask[:, :-1].clone()
                    mask[0] = False

                    sorted_probs[mask] = 0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                    out_index = torch.multinomial(sorted_probs, num_samples=1)
                    out_token = torch.gather(sorted_indexes, -1, out_index)

                if mode == "greedy" or mode == "greedy-instruct":
                    out_token = torch.argmax(out_probs, dim=1, keepdim=True) #greedy sampling

                else:
                    out_token = torch.multinomial(out_probs, num_samples=1)  # sample from the probabilities

                final_tokens = torch.cat((final_tokens, out_token), dim=-1)

                print(tokenizer.decode(out_token[0].tolist()), flush=True, end="")

                #if i % 50 == 0:
                #    print(out_probs[:, 50256])

                if out_token.item() == eot:
                    final_tokens = final_tokens[:, :-1]
                    break
                elif (out_token.item() == 50257 or out_token.item() == 50258) and (mode == "instruct" or mode == "greedy-instruct"):
                    break

            #final_tokens = torch.cat((final_tokens, torch.tensor(tokenizer.encode("\n")).unsqueeze(0).to(device)), dim=-1)
            print("\n")

        if in_text == "<|user|>\nbye\n<|assistant|>\n":
            final_tokens = torch.empty((1, 0), dtype=torch.long, device=device)
            print("bye")

    print(tokenizer.decode(final_tokens[0].tolist()))
    if outfile is not None:
        with open(outfile, "w") as f:
            f.write(tokenizer.decode(final_tokens[0].tolist()))

