from torchvision import transforms
from torch.utils.data import DataLoader
from utils import load_config
from data_cleaner import CarsDataset, create_image_dictionary
from torchvision import transforms

config = load_config()

train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

csv_path = config['car_csv']

# Create dataset instances for training and testing.
train_dataset = CarsDataset(config['train_data_path'], csv_path, transform=train_transforms)
test_dataset  = CarsDataset(config['test_data_path'], csv_path, transform=test_transforms)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
