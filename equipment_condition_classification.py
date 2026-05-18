"""Generated from Jupyter notebook: Equipment Condition Classification

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib as mpl
import torch
import torch.nn as nn
import torch.optim as optim


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1), nn.ReLU(), nn.Flatten(), nn.Linear(8 * 16 * 16, 2)
        )

    def forward(self, x):
        return self.net(x)


def main():
    m = CNN()
    opt = optim.Adam(m.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(10):
        opt.zero_grad()
        out = m(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
    acc = (m(X).argmax(1) == y).float().mean().item()
    print("Accuracy", round(acc, 2))


def main_alt() -> None:
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["figure.dpi"] = 120
    X = torch.rand(100, 1, 16, 16)
    (X.mean(dim=(1, 2, 3)) > 0.5).long()
    main()


if __name__ == "__main__":
    main()
