import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from utils import load_config  # adjust if needed

class CarsDataset(Dataset):
    def __init__(self, directory, csv_path, transform=None):

        self.directory = directory
        self.transform = transform

        # Load CSV into a pandas DataFrame

        # Build a dictionary mapping filename to label
        self.image_label_dict = create_image_dictionary(csv_path)
        
        # Build a list of image file paths (only .jpg images)
        self.image_paths = [
            os.path.join(directory, f) 
            for f in os.listdir(directory) 
            if f.lower().endswith('.jpg')
        ]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # Get the filename (without the directory path)
        fname = os.path.basename(img_path)
        # Retrieve the label from our dictionary; if not found, return -1 (or handle as needed)
        label = self.image_label_dict.get(fname, -1)
        return img, label


def create_image_dictionary(csv_path):
    # Loads the CSV file and returns a dictionary mapping filename to label.
    df = pd.read_csv(csv_path)
    image_label_dict = dict(zip(df['fname'], df['class']))
    return image_label_dict
