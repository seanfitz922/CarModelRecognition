import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_config, build_records
from data_loader import train_loader, val_loader
from train import train_epoch
from evaluate import evaluate_model
from model import MultiTaskResNet50

def main():
    config = load_config()
    
    json_dict_path = config['json_dict_path']
    train_txt_path = config['train_txt_path']
    
    # Build training records from the JSON and train.txt.
    train_records = build_records(json_dict_path, train_txt_path)
    all_models = {record["model_id"] for record in train_records}
    num_model_classes = len(all_models)

    # Derive the number of unique year classes from the train records.
    unique_years = {(record["year"]) for record in train_records}
    num_year_classes = len(unique_years)
    
    # print(f"Found {num_year_classes} unique year classes: {sorted(unique_years)}")
    
    model = MultiTaskResNet50(num_model_classes, num_year_classes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    num_epochs = config['num_epochs']
    print('-Training has started-')
    for epoch in range(num_epochs):
        train_loss, train_acc_model, train_acc_year = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc_model, val_acc_year = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Model Acc: {train_acc_model:.2f}%, Year Acc: {train_acc_year:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Model Acc: {val_acc_model:.2f}%, Year Acc: {val_acc_year:.2f}%")
        
        scheduler.step()

    final_model_path = os.path.join(config['model_save_path'], "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print("Training complete. Final model saved at:", final_model_path)

if __name__ == '__main__':
    main()
