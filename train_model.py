import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import tiktoken
from yummygpt import TransformerFinal, DataLoader
from datetime import datetime
import argparse
import math


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

parser = argparse.ArgumentParser()
parser.add_argument("-path", "--path", required=True)

args = parser.parse_args()
model_path = args.path #get model path

hyper_params = {
    #architectire
    "d_model": 512,
    "n_blocks": 12,
    "n_heads": 8,
    "vocab_size" : tokenizer.n_vocab,
    "weight_tying": True,
    "dropout": 0.1,

    #training
    "batch_size": 16,
    "context_len": 512,
    "steps": 20000,
    "T_max": 18000,
    "learn_rate": 3e-4,
    "train_split":0.9,
    "max_norm": 1.0,

    #evaluation
    "eval_interval": 50,
    "eval_batch_size": 8,

    #device settings
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "torch_seed": 1234
} #define hyper parameters

torch.manual_seed(hyper_params["torch_seed"]) #manual seed for reproducubility

if __name__ == "__main__":
    data = np.memmap(f"{model_path}.bin", dtype=np.uint16, mode="r") #only loads slices of the binary that is called
                                                                     #not directly a numpy array, use torch.from_numpy(np.array(data[slice])) in the dataloader
    train_data = data[:int(hyper_params["train_split"]*len(data))]   #make sure to decode as np.int64, which is torch.long
    eval_data = data[int(hyper_params["train_split"]*len(data)):] #get train and test splits

    trainloader = DataLoader(hyper_params["batch_size"], hyper_params["context_len"], train_data) #they only take slices of the data needed when the generate function is called
    evalloader = DataLoader(hyper_params["eval_batch_size"], hyper_params["context_len"], eval_data) #initialize data loaders

    m = TransformerFinal(hyper_params["d_model"], hyper_params["context_len"], hyper_params["n_heads"], hyper_params["n_blocks"], hyper_params["vocab_size"], hyper_params["dropout"], hyper_params["weight_tying"]).to(hyper_params["device"])
    optimizer = torch.optim.AdamW(m.parameters(), lr=hyper_params["learn_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyper_params["T_max"], eta_min=hyper_params["learn_rate"]*0.1) #initalize model, optimizer, and scheduler

    training_losses = []
    eval_losses = []
    avg_training_losses = []

    print("starting training....\n\n\n\n\n\n\n")
    start_time = datetime.now()

    try:
        for step in range(hyper_params["steps"]):
            inputs, targets = trainloader.generate() #generate inputs and targets from dataset
            inputs, targets = inputs.to(hyper_params["device"]), targets.to(hyper_params["device"])

            logits, loss = m(inputs, targets) #generate logits and loss from inputs and targets

            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=hyper_params["max_norm"])
            optimizer.step()
            scheduler.step() #backwards propagation and step the optimizer and scheduler

            training_losses.append(loss.item())
            avg_training_losses.append(sum(training_losses[:50])/50 if len(training_losses) > 50 else sum(training_losses[:len(training_losses)])/len(training_losses)) #log the losses

            if step % hyper_params["eval_interval"] == 0 or step == hyper_params["steps"] - 1:
                m.eval()

                now_time = datetime.now()

                with torch.no_grad():
                    eval_inputs, eval_targets = evalloader.generate()
                    eval_inputs, eval_targets = eval_inputs.to(hyper_params["device"]), eval_targets.to(hyper_params["device"])

                    eval_logits, eval_loss = m(eval_inputs, eval_targets)

                #print(f"step: {step}\ntraining loss:{loss.item():.2f}\ntesting loss: {eval_loss.item():.2f}\naverage training loss / 50 steps: {round(sum(training_losses[-50:]) / 50, 2) if len(training_losses) > 50 else round(sum(training_losses[-len(training_losses):]) / len(training_losses), 2)}\ntime / 50 epochs: {(now_time - start_time) / ((step / 50)) if step != 0 else 0}\nestimated time remaining: {(now_time - start_time) / step * (hyper_params['steps']) - (now_time - start_time) if step != 0 else 0} left\n" + "[" + "#" * round(step / hyper_params['steps']*50) + "-" * round(50 - step / hyper_params['steps']*50) + "]\n\x1b[7A")
                print("\x1b[7A")
                print(f"step: {step}")
                print(f"training loss:{loss.item():.2f}")
                print(f"testing loss: {eval_loss.item():.2f}")
                print(f"average training loss / 50 steps: {round(sum(training_losses[-50:]) / 50, 2) if len(training_losses) > 50 else round(sum(training_losses[-len(training_losses):]) / len(training_losses), 2)}")
                print(f"time / 50 epochs: {(now_time - start_time) / ((step / 50)) if step != 0 else 0}")
                print(f"estimated time remaining: {(now_time - start_time) / step * (hyper_params['steps']) - (now_time - start_time) if step != 0 else 0} left")
                print("[" + "#" * round(step / hyper_params['steps']*50) + "-" * round(50 - step / hyper_params['steps']*50) + "]") #print data

                eval_losses.append(eval_loss.item())

                m.train()

    except KeyboardInterrupt:
        print("stopping....")

    finally:
        end_time = datetime.now()

        print(f"total time training: {end_time - start_time}")
        print("saving model....")
        torch.save({"model_state_dict": m.state_dict(),
                    "hyper_params": hyper_params,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
        }, f"saved_models/{model_path}.pth")
        print("done") #save model to path file

        plt.plot(training_losses, color="red")
        plt.plot([50*i for i in range(len(eval_losses))], eval_losses, color="green")
        plt.plot(avg_training_losses, color="blue")

        plt.show() #plot losses

