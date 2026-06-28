import torch
import tiktoken
from yummygpt import YummyGPT
import json
import os, sys

model_path = input("model path?:")

if not os.path.exists(f"saved_models/{model_path}.pth"):
    print("Model path does not exist!\nexiting....")
    sys.exit(0)

with open("saved_models/config.json", "r", encoding="utf-8") as f:
    loaded_configs = json.load(f)

try:
    current_model_data = loaded_configs[model_path]
except KeyError:
    print("model config not saved!\nexiting.....")
    sys.exit(0)

d_model = current_model_data["d_model"]
sequence_length = current_model_data["sequence_length"]
n_heads = current_model_data["n_heads"]
n_blocks = current_model_data["n_blocks"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

tokenizer = tiktoken.get_encoding('gpt2')
eos_token = tokenizer.eot_token
vocab_size = tokenizer.n_vocab

m = YummyGPT(vocab_size, d_model, sequence_length=sequence_length, n_heads=n_heads, n_blocks=n_blocks)
m.load_state_dict(torch.load(f"saved_models/{model_path}.pth"))
m.to(device)
m.eval()

print(m)
print(f"\n{round(sum(p.numel() for p in m.parameters() if p.requires_grad)/ 1000000, 3)}M parameters\n")

while True:
    in_text = input("->:")
    if in_text == "exit":
        break
    input_ = torch.tensor(tokenizer.encode(in_text, allowed_special={"<|endoftext|>"}), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        output = m.generate(input_, max_new_tokens=500, eos_token=eos_token, temperature=1.0)
    print(tokenizer.decode(output[0].tolist()))



