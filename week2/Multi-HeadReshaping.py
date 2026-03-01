import torch
import torch.nn as nn

x = torch.randn(2, 5, 16)

print(x.shape)  # (B, T, d_model)

B, T, d_model = x.shape
heads = 4
d_k = d_model // heads

x = x.view(B, T, heads, d_k)
print(x.shape)  # (B, T, heads, d_k)

x = x.transpose(1, 2)
print(x.shape)  # (B, heads, T, d_k)

x = x.transpose(1, 2).contiguous().view(B, T, d_model)
print(x.shape)  # (B, T, d_model)