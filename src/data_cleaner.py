from PIL import Image
from torch.utils.data import Dataset

# class from RIPUPZ on kaggle: https://www.kaggle.com/code/ripupz/carsidentifier/notebook

class CarsDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx]["image_path"]
        label = self.dataframe.iloc[idx]["class_id"]  # Use integer labels

        # Open image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label
