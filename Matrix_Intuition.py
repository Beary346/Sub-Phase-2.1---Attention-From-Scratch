import torch
import torch.nn as nn

X = torch.randn(1, 5, 8)  # (batch_size, sequence_length, embedding_dim)

W_q = nn.Linear(8, 8)  # Query weight matrix
W_k = nn.Linear(8, 8)  # Key weight matrix
W_v = nn.Linear(8, 8)  # Value weight matrix

Q = W_q(X)
K = W_k(X)
V = W_v(X)

scores = Q @ K.transpose(-2, -1)
print("Scores shape:", scores.shape)  # (batch_size, sequence_length, sequence_length)
