import torch


def evaluate_accuracy(model, data_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = correct / total

    return accuracy


def calculate_sparsity(model, threshold=1e-2):

    total_connections = 0
    pruned_connections = 0

    for layer in [
        model.fc1,
        model.fc2,
        model.fc3
    ]:

        gates = torch.sigmoid(
            layer.gate_scores
        )

        total_connections += gates.numel()

        pruned_connections += (
            gates < threshold
        ).sum().item()

    sparsity = (
        pruned_connections
        / total_connections
        * 100
    )

    return sparsity