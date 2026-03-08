import torch
import torch.nn as nn

class DeepNoResidual(nn.Module):
    def __init__(self, depth=20, dim=64):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(depth)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x
    
class DeepResidual(nn.Module):
    def __init__(self, depth=20, dim=64):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(depth)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = x + torch.relu(layer(x))
        return x

class BlockWithNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.linear(x)
        return self.norm(x)


x = torch.randn(1, 64, requires_grad=True)

model = DeepNoResidual()
out = model(x)
loss = out.sum()
loss.backward()

print("No residual grad:", x.grad.norm())

model = DeepResidual()
out = model(x)
loss = out.sum()
loss.backward()

print("Residual grad:", x.grad.norm())

model = BlockWithNorm(64)
out = model(x)
loss = out.sum()
loss.backward()
print("Block with norm grad:", x.grad.norm())