import torch
import math

def positional_encoding(seq_len, d_model):

    position = torch.arange(seq_len).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
    )

    pe = torch.zeros(seq_len, d_model)

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

import matplotlib.pyplot as plt

pe = positional_encoding(100, 32)

plt.imshow(pe.T, aspect="auto")
plt.xlabel("Position")
plt.ylabel("Dimension")
plt.colorbar()
plt.show()