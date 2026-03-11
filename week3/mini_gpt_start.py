import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList(
            [MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)]
        )

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        print("Input shape:", x.shape)

        tok_emb = self.token_embedding(x)
        print("Token embedding shape:", tok_emb.shape)

        positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        print("Positions shape:", positions.shape)

        pos_emb = self.position_embedding(positions)
        print("Position embedding shape:", pos_emb.shape)

        x = tok_emb + pos_emb
        print("Input to blocks shape:", x.shape)

        for block in self.blocks:
            x = block(x)
        print("Output from block shape:", x.shape)

        logits = self.lm_head(x)
        return logits
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.dmodel = d_model
        self.num_heads = num_heads
        self.dhead = self.dmodel // self.num_heads

        assert self.dmodel % self.num_heads == 0, "dmodel must be divisible by num_heads"

        self.Wq = nn.Linear(self.dmodel, self.dmodel)
        self.Wk = nn.Linear(self.dmodel, self.dmodel)
        self.Wv = nn.Linear(self.dmodel, self.dmodel)
        self.Wo = nn.Linear(self.dmodel, self.dmodel)
    
    def forward(self, x):
        amount, rows, _ = x.shape
        mask = self.mask(rows).to(x.device)

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        Q = Q.view(amount, rows, self.num_heads, self.dhead).transpose(1, 2)
        K = K.view(amount, rows, self.num_heads, self.dhead).transpose(1, 2)
        V = V.view(amount, rows, self.num_heads, self.dhead).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.dhead)
        masked_scores = scores.masked_fill(mask==0, -1e9)
        masked_weights = F.softmax(masked_scores, dim=-1)

        output = masked_weights @ V
        output = output.transpose(1,2).contiguous().view(amount, rows, self.dmodel)

        return self.Wo(output)

    def mask(self,size):
        mask = torch.tril(torch.ones(size, size))
        return mask
    
vocab_size = 65
d_model = 64
num_heads = 8
num_layers = 2
max_seq_len = 128
model = TinyGPT(vocab_size, d_model, num_heads, num_layers, max_seq_len)
x = torch.randint(0, vocab_size, (4, 16))
logits = model(x)
print("Final logits shape:", logits.shape)