def train_epoch(model, train_loader, criterion, optimizer, train_dataset):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()             # Reset gradients
        outputs = model(inputs)           # Forward pass
        loss = criterion(outputs, labels) # Compute loss
        loss.backward()                   # Backward pass (compute gradients)
        optimizer.step()                  # Update weights

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    return epoch_loss
