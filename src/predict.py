import torch
from PIL import Image
from torchvision import transforms
from utils import load_config
import pandas as pd

# Example definitions for load_model() and preprocess_image()
config = load_config()

def preprocess_image(img):
    # If img is a file path (string or bytes), open it;
    # if it's already a PIL Image, use it directly.
    if isinstance(img, (str, bytes)):
        image = Image.open(img).convert("RGB")
    elif isinstance(img, Image.Image):
        image = img.convert("RGB")
    else:
        raise ValueError("Invalid image input. Expected a file path or a PIL Image.")

    test_transforms = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = test_transforms(image).unsqueeze(0)  # Add batch dimension

    return input_tensor

def load_combined_mapping(csv_path):
    df = pd.read_csv(csv_path)
    class_mapping = df[['class_id', 'true_class_name']].drop_duplicates().sort_values('class_id')
    return class_mapping

def get_model_name(class_mapping, class_index):
    return class_mapping[class_mapping['class_id'] == class_index]['true_class_name'].values[0]


def predict_image(img, model, device):
    input_tensor = preprocess_image(img).to(device)
    
    with torch.no_grad():
        output_model = model(input_tensor)
        _, predicted = output_model.max(1)
        class_index = predicted.item()

    class_mapping = load_combined_mapping(config['train_csv_path'])
    model_name = get_model_name(class_mapping, class_index)
    
    return model_name


