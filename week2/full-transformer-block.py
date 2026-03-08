import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dmodel = 36
        self.num_heads = 6
        self.dhead = self.dmodel // self.num_heads

        assert self.dmodel % self.num_heads == 0, "dmodel must be divisible by num_heads"

        self.Wq = nn.Linear(self.dmodel, self.dmodel)
        self.Wk = nn.Linear(self.dmodel, self.dmodel)
        self.Wv = nn.Linear(self.dmodel, self.dmodel)
        self.Wo = nn.Linear(self.dmodel, self.dmodel)
    
    def forward(self, x):
        amount, rows, cols = x.shape
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
    
class FeedForward(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear1 = nn.Linear(36, 36*4)
        self.linear2 = nn.Linear(36*4, 36)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        return x
    
x = torch.rand(1, 10, 36)
x.requires_grad_()

mha = MultiHeadAttention()
ffn = FeedForward()
LayerNorm = nn.LayerNorm(36)

x = x + mha(x)
x = LayerNorm(x) 
x = x + ffn(x)
x = LayerNorm(x)

print(x.shape)
print(x)