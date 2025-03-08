import torch

def evaluate_model(model, test_loader, criterion, test_dataset):
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    epoch_test_loss = test_loss / len(test_dataset)
    epoch_test_acc = correct / total

    return epoch_test_loss, epoch_test_acc