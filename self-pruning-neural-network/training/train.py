def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    lambda_value,
    device
):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        classification_loss = criterion(
            outputs,
            labels
        )

        sparsity_loss = model.sparsity_loss()

        total_loss = (
            classification_loss
            + lambda_value * sparsity_loss
        )

        total_loss.backward()

        optimizer.step()

        running_loss += total_loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        running_correct += (
            predictions == labels
        ).sum().item()

        total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = running_correct / total_samples

    return epoch_loss, epoch_accuracy