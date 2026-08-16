import torch.nn as nn

from .prunable_linear import PrunableLinear

class SelfPruningNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = PrunableLinear(3 * 32 * 32, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def sparsity_loss(self):
        return (
            self.fc1.sparsity_loss()
            + self.fc2.sparsity_loss()
            + self.fc3.sparsity_loss()
        )