import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch
import torch.nn as nn

from models.network import SelfPruningNetwork
from main import get_dataloaders
from training.train import train_one_epoch
from training.evaluate import (
    evaluate_accuracy,
    calculate_sparsity
)
from utils.metrics import get_gate_statistics
from utils.plot_gate_distribution import plot_gate_distribution


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    train_loader, test_loader = get_dataloaders()

    # Candidate lambda
    lambda_value = 0.005

    model = SelfPruningNetwork().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    num_epochs = 5

    print(
        f"\nTraining candidate model "
        f"with lambda = {lambda_value}"
    )

    for epoch in range(num_epochs):

        loss, accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            lambda_value,
            device
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {loss:.4f} "
            f"Accuracy: {accuracy * 100:.2f}%"
        )

    # Evaluation
    test_accuracy = evaluate_accuracy(
        model,
        test_loader,
        device
    )

    sparsity = calculate_sparsity(
        model,
        threshold=1e-2
    )

    print("\nFinal Results")
    print("----------------")
    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Sparsity: "
        f"{sparsity:.6f}%"
    )

    # Gate statistics
    print("\nGate Statistics")
    print("----------------")

    statistics = get_gate_statistics(model)

    for layer_name, stats in statistics.items():

        print(f"\n{layer_name}")

        print(
            f"Min: {stats['min']:.6f}"
        )

        print(
            f"Max: {stats['max']:.6f}"
        )

        print(
            f"Mean: {stats['mean']:.6f}"
        )

        print(
            f"Below 0.01: "
            f"{stats['below_0.01']} / "
            f"{stats['total']}"
        )

    # Plot gates
    plot_gate_distribution(model)

    # Save model
    os.makedirs("results", exist_ok=True)

    torch.save(
        model.state_dict(),
        "results/best_model.pth"
    )

    print(
        "\nModel saved to "
        "results/best_model.pth"
    )


if __name__ == "__main__":
    main()