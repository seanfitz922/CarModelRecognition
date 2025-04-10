from torch.utils.data import Subset, DataLoader
from torchvision import transforms
from utils import load_config, build_train_records, random_split
from data_class import CarsDataset

config = load_config()
batch_size = config['batch_size']
json_dict_path = config['json_dict_path']
train_txt_path = config['train_txt_path']

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

# Build training records as a list of dictionaries
train_records = build_train_records(json_dict_path, train_txt_path)

# Create indices for a random split
train_indices, val_indices = random_split(train_records)

# Create two separate dataset instances using the same records list but different transforms
full_train_dataset = CarsDataset(train_records, transform=train_transforms)
full_val_dataset = CarsDataset(train_records, transform=val_transforms)

# Create subsets for training and validation based on indices
train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_val_dataset, val_indices)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
