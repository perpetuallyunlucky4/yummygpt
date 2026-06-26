import torch
import tiktoken
from yummygpt import YummyGPT

sequence_length = 256
d_model = 256

model_path = input("model path?:")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

tokenizer = tiktoken.get_encoding('gpt2')
vocab_size = tokenizer.n_vocab

m = YummyGPT(vocab_size, d_model, sequence_length=sequence_length, n_heads=4, n_blocks=3)
m.load_state_dict(torch.load(f"{model_path}.pth"))
m.to(device)
m.eval()

print(m)

while True:
    in_text = input("->:")
    if in_text == "exit":
        break
    input_ = torch.tensor(tokenizer.encode(in_text), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        output = m.generate(input_, max_new_tokens=250)
    print(tokenizer.decode(output[0].tolist()))



