import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import math


class DataLoader():
    def __init__(self, batch_size, context_len, data):
        self.batch_size = batch_size
        self.context_len = context_len
        self.data = data

    def generate(self):
        indexes = torch.randint(0, len(self.data) - self.context_len, (self.batch_size,)) #generate random starts to sample from the data

        inputs = torch.stack([self.data[start:start+self.context_len] for start in indexes])
        targets = torch.stack([self.data[start+1:start+self.context_len+1] for start in indexes]) #stack all the inputs and targets into a (batch_size, context_len) shaped tensor

        return inputs, targets


class PositionalEmbeddings(nn.Module):
    def __init__(self, context_len, d_model, scale=10000):
        super().__init__()
        self.register_buffer("embedding", torch.zeros(context_len, d_model)) #register as a buffer so that it moves device with the model

        positions = torch.arange(context_len, dtype=torch.float32).view(context_len, 1) #arange of all the possible rows of each token

        denominators = scale ** (-torch.arange(0, d_model, 2, dtype=torch.float32)/d_model).view(1, d_model//2) #different frequencies of waves based on how deep you are through d_model

        self.embedding[:, 0::2] = torch.sin(positions @ denominators) #insert values into alternating d_model iteration
        self.embedding[:, 1::2] = torch.cos(positions @ denominators) #multiply each position by the corresponding denominator for each d_model iteration

    def forward(self, logits):
        return logits + self.embedding[:logits.shape[1], :] #add only the first sequence length size of the positional encoding to the logits


class MultiheadSelfAttention(nn.Module):
    def __init__(self, context_len, d_model, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_model // self.n_heads

        self.register_buffer("tril_mask", torch.tril(torch.ones(context_len, context_len))) #register as a buffer so it moves device with the model

        assert self.n_heads * self.d_head == d_model

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)#will be split into heads later

    def forward(self, logits):
        batch_size, context_len, d_model = logits.shape

        q = self.query(logits)
        k = self.key(logits)
        v = self.value(logits)#outputs shape (batch_size, context_length, d_model

        q = q.view(batch_size, context_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)#we split the output of the linear layer (d_model nodes) into heads. They do not talk to each other and we only consider each chunk of d_head output nodes from the input of all the d_model inputs
        k = k.view(batch_size, context_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)#next, swap the second and first index of each tensor to turn the n_heads dimension into a batch kinda thing, performing matrix mult on only the last 2 dimensions
        v = v.view(batch_size, context_len, self.n_heads, self.d_head).permute(0, 2, 1, 3)#produces batches of heads containing tensors shape (context_len, d_head)

        attention_scores = q @ k.transpose(-1, -2) #transpose to produce a correct multiplication of shape (batch_size, n_heads, context_len, context_len)
        attention_scores.masked_fill_(self.tril_mask[:context_len, :context_len]==0, float('-inf')) #mask the affinitiy matrix

        attention_probs = F.softmax(attention_scores/math.sqrt(self.d_head), dim=-1) #softmax to normalize and bring -inf values to zero

        attention = attention_probs @ v #matrix mult the masked affinity matrix with the value -- outputs shape (batch_size, n_heads, context_length, d_head)
        attention = attention.permute(0, 2, 1, 3).contiguous().view(batch_size, context_len, d_model) #swap the context_len and n_heads dimensions and make contiguous, then flatten the tensor into its initial shape (batch_size, context_len, d_model)

        logits = self.proj(attention) #run all the d_model paramers that are concatanated through all the heads through a final output projection to "summarize"

        return logits #(batch_size, context_len, d_model)

        #note: the matrix mults are all batch wise through batch_size and n_heads


class Block(nn.Module):
    def __init__(self, context_len, d_model, n_heads, dropout):
        super().__init__()

        self.att = MultiheadSelfAttention(context_len, d_model, n_heads) #init multihead attention
        self.ffw = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model)
        ) #feed forward neural network

        self.ln_att = nn.LayerNorm(d_model)
        self.ln_ffw = nn.LayerNorm(d_model) #pre layer norm used here

        self.drop = nn.Dropout(dropout) #only need one dropout, because it drops randomly every call

    def forward(self, logits):
        logits_att = self.att(self.ln_att(logits)) #get the output from self attention after normalizing
        logits = logits + self.drop(logits_att) #residual connection after dropout

        logits_ffw = self.ffw(self.ln_ffw(logits)) #get feed forward output after layer normalization
        logits = logits + self.drop(logits_ffw) #more residual connection after dropout

        return logits


class TransformerFinal(nn.Module):
    def __init__(self, d_model, context_len, n_heads, n_blocks, vocab_size, dropout=0.1, weight_tying=True):
        super().__init__()

        self.tok_embd = nn.Embedding(vocab_size, d_model)
        self.pos_embd = PositionalEmbeddings(context_len, d_model) #init embeddings

        self.blocks = nn.ModuleList(Block(context_len, d_model, n_heads, dropout) for i in range(n_blocks)) #stack n_blocks blocks together

        self.ln_final = nn.LayerNorm(d_model)

        self.line_out = nn.Linear(d_model, vocab_size) #final linear layer to transform from d_model to vocab_size

        if weight_tying:
            self.tok_embd.weight = self.line_out.weight #weight tying from the input embedding to the final output layer. Since they both encode and decode from vocab_size to d_model, we can make their weights equal

        self.init_weights()

    def init_weights(self, std=0.02, mean=0.0):
        print("initializing weights....")
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Embedding):
                nn.init.normal_(layer.weight, mean, std)
                print(f"initialized nn.Embedding {layer} {name}") #set embedding weights mean and standard deviation

            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean, std)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                print(f"initialized nn.Linear {layer} {name}") #set linear layers weights mean and standard deviation
        print("done")

    def forward(self, tokens, targets=None):
        logits = self.tok_embd(tokens)
        logits = self.pos_embd(logits) #encode tokens and position

        for block in self.blocks:
            logits = block(logits) #run logits through all the blocks sequentially

        logits = self.ln_final(logits)
        logits = self.line_out(logits) #final layer norm and neural network to expand out from d_model to vocab_size

        loss = None

        if targets is not None:
            batch_size, context_length, vocab_size = logits.shape
            logits = logits.view(batch_size * context_length, vocab_size)
            targets = targets.view(batch_size * context_length)
            loss = F.cross_entropy(logits, targets) #calculate loss from the targets if targets is not None

        return logits, loss

    def generate_tokens(self, tokens, max_iters=1000, temp=1.0, eos=None):
        for iter in range(max_iters):
            with torch.no_grad():
                logits, loss = self.forward(tokens) #run tokens through transformer
                logits = logits[:, -1, :] #take all the batches, the last token in the sequence, and all the outputs of the final token
                logits = logits/temp
                out_probs = F.softmax(logits, dim=1) #softmax the output to get probabililties
                out_token = torch.multinomial(out_probs, num_samples=1) #sample from the probabilities
                tokens = torch.cat((tokens, out_token), dim=1) #append the final token to all the tokens

                if eos is not None and out_token.item() == eos:
                    break #break if final token is an endoftext token

        return tokens #return the final list of tokens








