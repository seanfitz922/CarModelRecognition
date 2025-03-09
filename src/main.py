import torch, os
import torch.nn as nn
import torch.optim as optim
from model import CarModelCNN  # Adjust import as needed
from utils import load_config
from data_loader import train_loader, test_loader, train_dataset, test_dataset
from evaluate import evaluate_model
from train import train_epoch

def main():
    config = load_config()  # Expects keys: num_epochs, learning_rate, num_classes, model_save_path, etc.
    NUM_CLASSES = config['num_classes']
    num_epochs = config['num_epochs']
    
    # Create the model, loss function, and optimizer.
    model = CarModelCNN(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    best_val_acc = 0.0
    os.makedirs(config['model_save_path'], exist_ok=True)
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, train_dataset)
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, test_dataset)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Val Loss:   {val_loss:.4f}")
        print(f"    Val Acc:    {val_acc:.4f}")

        scheduler.step()
        
        # Save the model if validation accuracy improved.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(config['model_save_path'], "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print("    Best model saved!")
    
    # Save the final model
    final_model_path = os.path.join(config['model_save_path'], "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print("Training complete. Final model saved at:", final_model_path)

if __name__ == '__main__':
    main()