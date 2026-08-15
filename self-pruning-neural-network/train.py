import torch
import torch.nn as nn


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # without pruning, the weights and biases are initialized as usual
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features)
        )

        self.bias = nn.Parameter(
            torch.zeros(out_features)
        )
          
        # The gate scores are initialized randomly and will be learned during training

        self.gate_scores = nn.Parameter(
            torch.randn(out_features, in_features)
        )

    def forward(self, x):

        # Converting gate scores to values between 0 and 1
        gates = torch.sigmoid(self.gate_scores)

        # Applying  gates to weights
        pruned_weights = self.weight * gates

        # Normal linear operation
        output = x @ pruned_weights.T + self.bias

        return output
    # Sparsity Loss = sum(all gate values)
    def sparsity_loss(self):
        gates = torch.sigmoid(self.gate_scores)
        return gates.sum()
# complete network
class SelfPruningNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = PrunableLinear(3 * 32 * 32, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Flatten image
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

    
# testing
if __name__ == "__main__":

    model = SelfPruningNetwork()

    # Fake CIFAR-10 batch
    x = torch.randn(4, 3, 32, 32)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    sparsity = model.sparsity_loss()

    print("Sparsity loss:", sparsity.item())

    # Check all parameters
    print("\nModel parameters:")

    for name, parameter in model.named_parameters():
        print(name, parameter.shape, parameter.requires_grad)