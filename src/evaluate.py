from tqdm import tqdm

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_running_loss = 0.0
    correct_model = 0
    correct_year = 0
    total = 0

    # Wrap your DataLoader with tqdm to monitor progress
    for images, labels in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        model_labels = labels[0].to(device)
        year_labels = labels[1].to(device)
        
        model_pred, year_pred = model(images)
        loss_model = criterion(model_pred, model_labels)
        loss_year = criterion(year_pred, year_labels)
        loss = loss_model + loss_year
        
        val_running_loss += loss.item()
        _, pred_model = model_pred.max(1)
        _, pred_year = year_pred.max(1)
        correct_model += (pred_model == model_labels).sum().item()
        correct_year += (pred_year == year_labels).sum().item()
        total += images.size(0)

    val_loss = val_running_loss / len(val_loader)
    val_acc_model = 100 * correct_model / total
    val_acc_year = 100 * correct_year / total
    return val_loss, val_acc_model, val_acc_year
