import torch
import torch.nn.functional as F
import math

torch.manual_seed(0)

B, T, D = 1, 5, 8

Q = torch.randn(B, T, D)
K = torch.randn(B, T, D)
V = torch.randn(B, T, D)

def attention(Q, K, V, temperature=1.0):
    d_k = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    scores = scores / temperature
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights

for temp in [0.5, 1.0, 2.0]:
    _, weights = attention(Q, K, V, temperature=temp)
    print(f"\nTemperature: {temp}")
    print(weights.min(), weights.max())