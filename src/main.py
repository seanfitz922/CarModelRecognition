import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet50_Weights
from utils import load_config
from data_loader import train_loader, val_loader
from train import train_epoch
from evaluate import evaluate_model
from model import CarModelCNN

def main():
    config = load_config()
    
    # Load the pretrained ResNet50 model with default weights
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_classes = 196  # Adjust as needed for your dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    num_epochs = config['num_epochs']  # Adjust number of epochs as needed

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%")
        
        scheduler.step()  # Update learning rate

    final_model_path = os.path.join(config['model_save_path'], "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print("Training complete. Final model saved at:", final_model_path)

if __name__ == '__main__':
    main()
