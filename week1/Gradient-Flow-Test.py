import torch
import torch.nn as nn

X = torch.randn(1, 5, 8)  # (batch_size, sequence_length, embedding_dim)

W_q = nn.Linear(8, 8)  # Query weight matrix
W_k = nn.Linear(8, 8)  # Key weight matrix
W_v = nn.Linear(8, 8)  # Value weight matrix

Q = W_q(X)
K = W_k(X)
V = W_v(X)

Q.requires_grad_()
K.requires_grad_()
V.requires_grad_()

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    weights = torch.softmax(scores, dim=-1)
    return weights @ V

output = scaled_dot_product_attention(Q, K, V)
loss = output.sum()
loss.backward()