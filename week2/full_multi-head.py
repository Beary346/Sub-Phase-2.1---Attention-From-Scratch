import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, D = x.shape
        
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        mask = self.mask(T)
        
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        masked_scores = scores.masked_fill(mask == 0, float('-inf'))
        masked_weights = F.softmax(masked_scores, dim=-1)
        out = masked_weights @ V
        
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
        return self.W_o(out), scores, masked_scores
    
    def mask(self, size):
        return torch.tril(torch.ones(size, size))
    
# Example usage
x = torch.randn(4, 10, 25)  # (B, T, d_model)
mha = MultiHeadAttention(d_model=25, num_heads=5)
output, attention_weights, masked_attention_weights = mha.forward(x)
print(output.shape)  # (B, T, d_model)
#print(output)
#print(attention_weights)  # (B, num_heads, T, T)
#print(masked_attention_weights.shape)  # (B, num_heads, T, T)
#print(masked_attention_weights)  # (B, num_heads, T, T)