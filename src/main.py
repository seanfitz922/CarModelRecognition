import os
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from utils import load_config
from data_loader import train_loader, val_loader
from train import train_epoch
from evaluate import evaluate_model
from tqdm import tqdm

def main():
    config = load_config()
    learning_rate = config['learning_rate']
    
    # Load the pretrained ResNet50 model with default weights
    model = timm.create_model('efficientnetv2_rw_m', pretrained=True, num_classes=196)

    for param in model.parameters():
        param.requires_grad = True
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    num_epochs = config['num_epochs']  

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%")
        
        scheduler.step()  # Update learning rate

    final_model_path = os.path.join(config['model_save_path'], "final_model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print("Training complete. Final model saved at:", final_model_path)

if __name__ == '__main__':
    main()
