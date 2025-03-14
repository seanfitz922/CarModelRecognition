import pandas as pd
import random
from torch.utils.data import Subset
from torchvision import transforms
from utils import load_config, random_split
from data_cleaner import CarsDataset
from torch.utils.data import DataLoader

config = load_config()
batch_size = config['batch_size']

# Load CSVs
train_df = pd.read_csv(config['train_csv_path'])
test_df = pd.read_csv(config['test_csv_path'])

# Define separate transforms for training and validation
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Create indices for a random split
train_indices, val_indices = random_split(train_df)

# Create two separate dataset instances using the same CSV but different transforms
full_train_dataset = CarsDataset(train_df, transform=train_transforms)
full_val_dataset = CarsDataset(train_df, transform=val_transforms)

# Create subsets for training and validation based on indices
train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_val_dataset, val_indices)

# create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
