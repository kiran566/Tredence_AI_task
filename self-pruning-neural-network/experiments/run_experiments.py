
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import csv

import torch
import torch.nn as nn

from models.network import SelfPruningNetwork
from data.data_loaders import get_dataloaders
from training.train import train_one_epoch
from training.evaluate import (
    evaluate_accuracy,
    calculate_sparsity
)


# --------------------------------------------------
# Run one experiment for one lambda value
# --------------------------------------------------

def run_experiment(
    lambda_value,
    num_epochs,
    train_loader,
    test_loader,
    device
):

    print("\n" + "=" * 60)
    print(f"Starting experiment: lambda = {lambda_value}")
    print("=" * 60)

    # IMPORTANT:
    # Fresh model for every lambda experiment
    model = SelfPruningNetwork().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # -------------------------------
    # Training
    # -------------------------------

    for epoch in range(num_epochs):

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            lambda_value=lambda_value,
            device=device
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {train_loss:.4f} "
            f"Accuracy: {train_accuracy * 100:.2f}%"
        )

    # -------------------------------
    # Evaluation
    # -------------------------------

    test_accuracy = evaluate_accuracy(
        model,
        test_loader,
        device
    )

    sparsity = calculate_sparsity(
        model,
        threshold=1e-2
    )

    print("\nExperiment Result")
    print("-----------------")
    print(f"Lambda:        {lambda_value}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Sparsity:      {sparsity:.6f}%")

    return {
        "lambda": lambda_value,
        "test_accuracy": test_accuracy * 100,
        "sparsity": sparsity
    }


# --------------------------------------------------
# Save experiment results
# --------------------------------------------------

def save_results(results):

    os.makedirs("results", exist_ok=True)

    file_path = "results/lambda_results.csv"

    with open(
        file_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "lambda",
                "test_accuracy",
                "sparsity"
            ]
        )

        writer.writeheader()

        writer.writerows(results)

    print(f"\nResults saved to: {file_path}")


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

def main():

    # -------------------------------
    # Device
    # -------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    # -------------------------------
    # Load CIFAR-10
    # -------------------------------

    train_loader, test_loader = get_dataloaders(
        batch_size=128
    )

    # -------------------------------
    # Lambda values
    # -------------------------------

    lambda_values = [
        # 0.00001,
        # 0.0001,
        # 0.0005,
        # 0.001,
        0.001,
        0.005,
        0.01,
        0.05

    ]

    # -------------------------------
    # Number of epochs
    # -------------------------------

    num_epochs = 5

    results = []

    # -------------------------------
    # Run experiments
    # -------------------------------

    for lambda_value in lambda_values:

        result = run_experiment(
            lambda_value=lambda_value,
            num_epochs=num_epochs,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device
        )

        results.append(result)

    # -------------------------------
    # Print final comparison
    # -------------------------------

    print("\n\nFINAL LAMBDA COMPARISON")
    print("=" * 70)

    print(
        f"{'Lambda':<12}"
        f"{'Accuracy (%)':<18}"
        f"{'Sparsity (%)':<18}"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['lambda']:<12}"
            f"{result['test_accuracy']:<18.2f}"
            f"{result['sparsity']:<18.6f}"
        )

    # -------------------------------
    # Save CSV
    # -------------------------------

    save_results(results)


if __name__ == "__main__":
    main()