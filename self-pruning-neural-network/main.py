import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.network import SelfPruningNetwork
from training.train import train_one_epoch
from training.evaluate import evaluate_accuracy, calculate_sparsity
from utils.metrics import get_gate_statistics

def get_dataloaders(batch_size=128):

    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    train_loader, test_loader = get_dataloaders()

    model = SelfPruningNetwork().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    lambda_value = 0.0001
    num_epochs = 1

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

    test_accuracy = evaluate_accuracy(
        model,
        test_loader,
        device
    )

    sparsity = calculate_sparsity(model)

    print("\nFinal Evaluation")
    print("----------------")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Sparsity: {sparsity:.2f}%")
    gate_statistics = get_gate_statistics(model)

    print("\nGate Statistics")
    print("----------------")

    for layer_name, stats in gate_statistics.items():
        print(f"\n{layer_name}")

        print(f"Min gate: {stats['min']:.6f}")
        print(f"Max gate: {stats['max']:.6f}")
        print(f"Mean gate: {stats['mean']:.6f}")
        print(
           f"Gates < 0.01: "
           f"{stats['below_0.01']} / {stats['total']}"
        )



if __name__ == "__main__":
    main()