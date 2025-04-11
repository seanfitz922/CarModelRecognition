
# simple training loop
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_model = 0
    correct_year = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        # labels is a tuple: (model_labels, year_labels)
        model_labels = labels[0].to(device)
        year_labels = labels[1].to(device)

        optimizer.zero_grad()
        model_pred, year_pred = model(images)
        loss_model = criterion(model_pred, model_labels)
        loss_year = criterion(year_pred, year_labels)
        loss = loss_model + loss_year
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, pred_model = model_pred.max(1)
        _, pred_year = year_pred.max(1)
        correct_model += (pred_model == model_labels).sum().item()
        correct_year += (pred_year == year_labels).sum().item()
        total += images.size(0)
    
    train_loss = running_loss / len(train_loader)
    train_acc_model = 100 * correct_model / total
    train_acc_year = 100 * correct_year / total
    return train_loss, train_acc_model, train_acc_year
