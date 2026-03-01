import torch
import torch.nn as nn
import math

torch.manual_seed(10)

A = 1
B = 7
C = 4

X = torch.randn(A, B, C)  # (batch_size, sequence_length, embedding_dim)

W_q = nn.Linear(C, C)
W_k = nn.Linear(C, C)
W_v = nn.Linear(C, C)

Q = W_q(X)
K = W_k(X)
V = W_v(X)

scores = Q @ K.transpose(-2, -1) / math.sqrt(C)
weights = nn.functional.softmax(scores, dim=-1)

print("Attention Weights Without Masking:")
print(weights[0])

def causal_mask(size):
    return torch.tril(torch.ones(size, size))

mask = causal_mask(B)
print("Mask:")
print(mask)

scores_masked = scores.masked_fill(mask == 0, float('-inf'))
weights_masked = nn.functional.softmax(scores_masked, dim=-1)

print("Attention Weights WITH mask:")
print(weights_masked[0])