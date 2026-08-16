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
