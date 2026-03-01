import torch.nn as nn
import torch

X = torch.randn(1, 5, 8)

W_Q = nn.Linear(8, 8)
W_K = nn.Linear(8, 8)
W_V = nn.Linear(8, 8)

Q = W_Q(X)
K = W_K(X)
V = W_V(X)

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    weights = torch.softmax(scores, dim=-1)
    return weights @ V