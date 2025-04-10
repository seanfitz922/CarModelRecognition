import torch
import torch.nn as nn
from utils import load_config, build_records
from evaluate import evaluate_model
from model import MultiTaskResNet50
from torch.utils.data import DataLoader
from data_class import CarsDataset
from torchvision import transforms

# PARALELLIZE ME YOU LAZY RAT

def test_model():
    config = load_config()

    json_dict_path = config['json_dict_path']
    test_txt_path = config['test_txt_path']
    final_model_path = config['final_model_path']

    test_records = build_records(json_dict_path, test_txt_path)

    all_models = {record["model_id"] for record in test_records}
    num_model_classes = len(all_models)
    print(all_models)

    unique_years = {(record["year"]) for record in test_records}
    num_year_classes = len(unique_years)
    print(num_model_classes, num_year_classes)

    model = MultiTaskResNet50(num_model_classes, num_year_classes)

    model.load_state_dict(torch.load(final_model_path))

    test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = CarsDataset(test_records, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loss, test_acc_model, test_acc_year = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Model Accuracy: {test_acc_model:.2f}%, Year Accuracy: {test_acc_year:.2f}%")



if __name__ == '__main__':
    test_model()


