from PIL import Image
from torch.utils.data import Dataset

class CarsDataset(Dataset):
    def __init__(self, train_records, transform=None):
        # train_records is a list of dictionaries as produced by build_train_records
        self.data = train_records
        self.transform = transform

        # Build mapping for year labels (preserving non-numeric entries like "unknown")
        all_years = {record["year"] for record in self.data}
        all_years = sorted(list(all_years), key=lambda x: (x == "unknown", x))
        self.year2index = {year: idx for idx, year in enumerate(all_years)}
        # print("Year mapping:", self.year2index)

        # Build mapping for model labels:
        # We'll treat the raw model_id (stored as a string) as a key and assign contiguous indices.
        all_models = {record["model_id"] for record in self.data}
        # Sort by numeric value (if they are numeric strings)
        all_models = sorted(list(all_models), key=lambda x: int(x))
        self.model2index = {model: idx for idx, model in enumerate(all_models)}
        # print("Model mapping:", self.model2index)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        image_path = record["image_path"]
        
        # Load the image on demand
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Use the mappings to convert raw labels to contiguous indices
        model_label = self.model2index[record["model_id"]]
        year_label = self.year2index[record["year"]]
        
        return image, (model_label, year_label)
