import torch
import matplotlib.pyplot as plt


def get_all_gates(model):

    all_gates = []

    for layer in [
        model.fc1,
        model.fc2,
        model.fc3
    ]:

        gates = torch.sigmoid(
            layer.gate_scores
        )

        all_gates.append(
            gates.detach().cpu().flatten()
        )

    return torch.cat(all_gates)


def plot_gate_distribution(model):

    gates = get_all_gates(model)

    plt.figure(figsize=(8, 5))

    plt.hist(
        gates.numpy(),
        bins=50
    )

    plt.axvline(
        0.01,
        linestyle="--",
        label="Pruning threshold"
    )

    plt.xlabel("Gate value")
    plt.ylabel("Number of connections")
    plt.title("Gate Value Distribution")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/gate_distribution.png"
    )

    plt.show()