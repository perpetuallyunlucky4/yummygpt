import torch
import tiktoken
from datetime import datetime
from yummygpt import YummyGPT, DataLoader
import matplotlib.pyplot as plt
import os, sys
import json

training_batch_size = 8
testing_batch_size = 8
epochs = 30000
T_max = 29000
d_model = 384
sequence_length = 256
n_heads = 4
n_blocks = 6
model_exists = False

model_path = input("model path?:")

if os.path.exists(f"saved_models/{model_path}.pth"):
    model_exists = True
    print("found existing model path, loading....")
    try:
        with open("saved_models/config.json", "r", encoding="utf-8") as f:
            loaded_configs = json.load(f)

    except FileNotFoundError:
        print("model config not saved!\nexiting....")
        sys.exit(0)

    current_model_data = loaded_configs[model_path]

    d_model = current_model_data["d_model"]
    sequence_length = current_model_data["sequence_length"]
    n_heads = current_model_data["n_heads"]
    n_blocks = current_model_data["n_blocks"]

print(f"d_model: {d_model}\nsequence_length: {sequence_length}\nn_heads: {n_heads}\nn_blocks: {n_blocks}\n")

if not os.path.exists(f"{model_path}.txt"):
    print("Training path does not exist!\nexiting....")
    sys.exit(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

with open(f"{model_path}.txt", "r") as f:
    text = f.read()

tokenizer = tiktoken.get_encoding('gpt2')
eos_token = tokenizer.eot_token

vocab_size = tokenizer.n_vocab

print("encoding data....")
data = torch.tensor(tokenizer.encode(text, allowed_special={"<|endoftext|>"}), dtype=torch.long, device=device)
print("done\n")

training_data = data[:int(len(data)*0.9)]
testing_data = data[int(len(data)*0.9):]

training_loader = DataLoader(training_data, training_batch_size, sequence_length)
testing_loader = DataLoader(testing_data, testing_batch_size, sequence_length)

m = YummyGPT(vocab_size, d_model, sequence_length=sequence_length, n_heads=n_heads, n_blocks=n_blocks)
if model_exists:
    m.load_state_dict(torch.load(f"saved_models/{model_path}.pth"))
m.to(device)
if device == "cuda":
    print("compiling...")
    m = torch.compile(m)
    print("done")

lr = 2e-4
optim = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=T_max, eta_min=lr*0.1)

print("starting training.....\n")
print(f"vocab_size: {vocab_size}\nepochs: {epochs}")

training_loss = []
testing_loss = []
avg_losses = []

start_time = datetime.now()

try:
    for epoch in range(epochs):
        xb, yb = training_loader.get_batch()

        logits, loss = m.forward(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)

        optim.step()
        scheduler.step()

        training_loss.append(loss.item())
        avg_losses.append(sum(training_loss[-50:]) / 50 if len(training_loss) > 50 else sum(training_loss[-len(training_loss):]) / len(training_loss))

        if epoch % 50 == 0 or epoch == epochs - 1:
            m.eval()
            now_time = datetime.now()
            with torch.no_grad():
                xvb, yvb = testing_loader.get_batch()
                logits_test, loss_test = m.forward(xvb, yvb)
            print(f"epoch: {epoch}\ntraining loss:{loss.item():.2f}\ntesting loss: {loss_test.item():.2f}\naverage training loss / 50 epochs: {round(sum(training_loss[-50:])/50, 2) if len(training_loss) > 50 else round(sum(training_loss[-len(training_loss):])/len(training_loss), 2)}\ntime: {now_time - start_time} since {start_time}\ntime / 50 epochs: {(now_time - start_time)/((epoch/50)) if epoch != 0 else 0}\nestimated time remaining: {(now_time - start_time)/((epoch/50))*(epochs/50) if epoch != 0 else 0} left")
            testing_loss.append(loss_test.item())
            m.train()

except KeyboardInterrupt:
    end_time = datetime.now()
    print(f"total time training: {end_time - start_time}\n")

    torch.save(m.state_dict(), f"saved_models/{model_path}.pth")
    print(f"saved model to 'saved_models/{model_path}.pth'\n--------------------------------------------\n")

    if not model_exists:
        with open("saved_models/config.json", "r", encoding="utf-8") as f:
            loaded_configs = json.load(f)
    
        new_config = {
            model_path: {
                "d_model": d_model,
                "sequence_length": sequence_length,
                "n_heads": n_heads,
                "n_blocks": n_blocks
            },
        }
    
        loaded_configs.update(new_config)
    
        with open("saved_models/config.json", "w", encoding="utf-8") as f:
            json.dump(loaded_configs, indent=4, fp=f)
    
        print("saved model config to 'saved_models/config.json'\n--------------------------------------------\n")

    training_loss_epochs = [i for i in range(len(training_loss))]
    testing_loss_epochs = [50*i for i in range(len(testing_loss))]

    plt.plot(training_loss_epochs, training_loss, label="training loss")
    plt.plot(testing_loss_epochs, testing_loss, label="testing loss")
    plt.plot(training_loss_epochs, avg_losses, label="average loss/50")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend()
    plt.show()

    print("ending program.....")
    sys.exit(0)

end_time = datetime.now()
print(f"total time training: {end_time - start_time}\n")

torch.save(m.state_dict(), f"saved_models/{model_path}.pth")
print(f"saved model to 'saved_models/{model_path}.pth'\n--------------------------------------------\n")

if not model_exists:
    with open("saved_models/config.json", "r", encoding="utf-8") as f:
        loaded_configs = json.load(f)
    
    new_config = {
        model_path : {
            "d_model" : d_model,
            "sequence_length" : sequence_length,
            "n_heads" : n_heads,
            "n_blocks" : n_blocks
        },
    }
    
    loaded_configs.update(new_config)
    
    with open("saved_models/config.json", "w", encoding="utf-8") as f:
        json.dump(loaded_configs, indent=4, fp=f)
    
    print("saved model config to 'saved_models/config.json'\n--------------------------------------------\n")

training_loss_epochs = [i for i in range(epochs)]
testing_loss_epochs = [50*i for i in range(int((epochs+50)/50))]

plt.plot(training_loss_epochs, training_loss, label="training loss")
plt.plot(testing_loss_epochs, testing_loss, label="testing loss")
plt.plot(training_loss_epochs, avg_losses, label="average loss/50")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend()
plt.show()

while True:
    in_text = input("->:")
    if in_text == "exit":
        break
    input_ = torch.tensor(tokenizer.encode(in_text, allowed_special={"<|endoftext|>"}), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        output = m.generate(input_, max_new_tokens=250, eos_token=None, temperature=0.7) #set eos token to none for testing
    print(tokenizer.decode(output[0].tolist()))





