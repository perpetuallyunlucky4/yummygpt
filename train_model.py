import torch
import tiktoken
from datetime import datetime
from yummygpt import YummyGPT, DataLoader

training_batch_size = 16
testing_batch_size = 8
sequence_length = 256
d_model = 256

model_path = input("model path?:")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

with open(f"{model_path}.txt", "r") as f:
    text = f.read()

tokenizer = tiktoken.get_encoding('gpt2')
vocab_size = tokenizer.n_vocab

data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)

training_data = data[:int(len(data)*0.8)]
testing_data = data[int(len(data)*0.8):]

training_loader = DataLoader(training_data, training_batch_size, sequence_length)
testing_loader = DataLoader(testing_data, testing_batch_size, sequence_length)

m = YummyGPT(vocab_size, d_model, sequence_length=sequence_length, n_heads=4, n_blocks=3)
m.to(device)

lr = 4e-4
optim = torch.optim.AdamW(m.parameters(), lr=lr)

epochs = 1000

print("starting training.....\n")
print(f"vocab_size at {vocab_size}\n\n")

start_time = datetime.now()
for epoch in range(epochs):
    xb, yb = training_loader.get_batch()

    logits, loss = m.forward(xb, yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()
    if epoch % 100 == 0 or epoch == epochs - 1:
        m.eval()
        now_time = datetime.now()
        with torch.no_grad():
            xvb, yvb = testing_loader.get_batch()
            logits_test, loss_test = m.forward(xvb, yvb)
            print(f"epoch: {epoch}\ntraining loss:{loss}\ntesting loss: {loss_test}\ntime: {now_time - start_time} since {start_time}\naverage time / 100 epochs:{(now_time - start_time) / (epoch + 1)}\n")
        m.train()

end_time = datetime.now()
print(f"total time training: {end_time - start_time}\n")

torch.save(m.state_dict(), f"{model_path}.pth")
print(f"saved model to '{model_path}.pth'\n--------------------------------------------\n\n")

while True:
    in_text = input("->:")
    if in_text == "exit":
        exit()
    input_ = torch.tensor(tokenizer.encode(in_text), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        output = m.generate(input_, max_new_tokens=250)
    print(tokenizer.decode(output[0].tolist()))





