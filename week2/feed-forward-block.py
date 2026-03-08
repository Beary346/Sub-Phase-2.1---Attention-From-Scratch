import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.net(x)
        return x
    
# Example usage
x = torch.randn(2, 6, 16)  # (batch, seq_len, d_model)

ff = FeedForward(d_model=16)
x.requires_grad_()

optimizer = torch.optim.Adam(ff.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for _ in range(500):
    optimizer.zero_grad()
    out = ff(x)
    loss = criterion(out, target=torch.zeros_like(out))
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        print(f"Loss: {loss.item():.4f}")
