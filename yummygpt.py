import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class DataLoader:
    def __init__(self, tokens, batch_size, sequence_length):
        self.tokens = tokens
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.current_pos = 0

    def get_batch(self):
        b, c = self.batch_size, self.sequence_length
        max_start = len(self.tokens) - c - 1

        starts = torch.randint(0, max_start, (b, ), device=self.tokens.device)

        x = torch.stack([
            self.tokens[s : s + c] for s in starts
        ])

        y = torch.stack([
            self.tokens[s + 1: s + c + 1] for s in starts
        ])

        return x, y

class PositionalEncodings(nn.Module):
    def __init__(self, sequence_length, d_model):
        super().__init__()
        pe = torch.zeros(sequence_length, d_model)
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)#dimensions sequence length, 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # dimensions d_model/2 , 1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#shape 1, sequence_length, d_model

        self.register_buffer('pe', pe)

    def forward(self, logits):
        return logits + self.pe[:, :logits.size(1), :]

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, inputs):
        batch_size, sequence_length, d_model = inputs.shape

        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_model)
        mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).bool().to(inputs.device)
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention = torch.matmul(attention_probs, v)

        out = self.out(attention)

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert (self.n_heads * self.head_dim == d_model)

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, inputs):
        batch_size, sequence_length, d_model = inputs.shape

        q = self.query(inputs).view(batch_size, sequence_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(inputs).view(batch_size, sequence_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(inputs).view(batch_size, sequence_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).bool().to(inputs.device)
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))

        attention_probs = torch.softmax(attention_scores, dim=-1)

        attention = torch.matmul(self.drop(attention_probs), v)
        attention = attention.permute(0, 2, 1, 3).contiguous()
        attention = attention.view(batch_size, sequence_length, d_model)

        out = self.out(attention)

        return out

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.att = MultiHeadAttention(d_model, n_heads)
        self.fcn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(0.1)

    def forward(self, logits):
        att_logits = self.att.forward(logits)
        adn_logits = self.ln1(logits + att_logits)
        logits = self.drop(adn_logits)
        logits = self.fcn(logits)
        logits = self.ln2(logits + adn_logits)

        return logits

class YummyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, sequence_length, n_heads=4, n_blocks=2):
        super().__init__()

        self.sequence_length = sequence_length
        
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = PositionalEncodings(sequence_length, d_model)
        self.blocks = nn.ModuleList([GPTBlock(d_model, n_heads) for i in range(n_blocks)])
        self.fl = nn.Linear(d_model, vocab_size)

        self.fl.weight = self.wte.weight

    def forward(self, inputs, targets=None):
        logits = self.wte(inputs) #dimensions batch size * sequence length * d_model
        logits = self.wpe.forward(logits)
        for block in self.blocks:
            logits = block.forward(logits)
        logits = self.fl(logits)
        loss = None
        if targets is not None:
            batch_size, sequence_length, d_model = logits.shape
            logits = logits.view(batch_size * sequence_length, d_model)
            targets = targets.view(batch_size * sequence_length)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, inputs, max_new_tokens, eos_token=None, temperature=1.0):
        if temperature == 0:
            print("\ntemperature set to 1.0\n")
            temperature = 1.0
        for i in range(max_new_tokens):
            inputs_cropped = inputs[:, -self.sequence_length:]
            
            logits, loss = self.forward(inputs_cropped)
            logits = logits[:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=1)
            next_token = torch.multinomial(probs, num_samples=1)
            inputs = torch.cat([inputs, next_token], dim=1)

            if eos_token is not None and next_token.item() == eos_token:
                break
        return inputs



