import torch
import torch.nn as nn
import math

class SelfAttention():
    def __init__(self, embedding_dim):
        self.W_q = nn.Linear(embedding_dim, embedding_dim)
        self.W_k = nn.Linear(embedding_dim, embedding_dim)
        self.W_v = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, X, temperature=1.0):
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        d_k = Q.size(-1)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
        scores = scores / temperature
        mask = self.casual_mask(Q.size(-2))

        scores_masked = scores.masked_fill(mask==0, float('-inf'))
        weights_masked = nn.functional.softmax(scores_masked, dim=-1)

        return weights_masked, weights_masked @ V
    
    def casual_mask(self, size):
        return torch.tril(torch.ones(size, size))
    
# Example usage
torch.manual_seed(10)

X = torch.randn(1, 6, 9)
attention = SelfAttention(X.size(-1))

masked_weights, output = attention.forward(X)
print(masked_weights, output)