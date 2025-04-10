import json
import torch
from PIL import Image
from torchvision import transforms
from model import MultiTaskResNet50
from utils import load_config

# Example definitions for load_model() and preprocess_image()
config = load_config()
def load_model():
    # Use the same numbers you trained with.
    model = MultiTaskResNet50(431, 13)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(config['final_model_path'], map_location=device))
    model.to(device)
    model.eval()
    return model, device

# Define your test transforms – they should be the same as used in training/testing.
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = test_transforms(image).unsqueeze(0)  # Add batch dimension

    return input_tensor

def load_combined_mapping(json_path):
    """Load combined_mapping.json which now contains both the data and metadata."""
    with open(json_path, "r") as f:
        mapping_json = json.load(f)
    return mapping_json

def build_index2model(mapping_data):
    """
    Build a reverse mapping (index2model) from your combined mapping data.
    The expected structure is:
      mapping_data[maker][model][year] = list of image records
    Here, the unique model IDs are the keys at the second level.
    """
    unique_models = set()
    for maker in mapping_data:
        for model in mapping_data[maker].keys():
            unique_models.add(model)
    # IMPORTANT: Make sure the sorting here exactly matches what was done during training.
    sorted_models = sorted(list(unique_models), key=lambda x: int(x))
    model2index = {model: idx for idx, model in enumerate(sorted_models)}
    index2model = {idx: model for model, idx in model2index.items()}
    return index2model

def predict_image(img):
    # Load model and preprocess image.
    model, device = load_model()
    input_tensor = preprocess_image(img)
    
    with torch.no_grad():
        output_model, output_year = model(input_tensor.to(device))
        # Get the predicted indices.
        _, pred_model_idx = torch.max(output_model, 1)
        _, pred_year_idx = torch.max(output_year, 1)

    # Load the combined mapping JSON that includes metadata.
    mapping_json = load_combined_mapping(config['json_dict_path'])
    
    # The combined mapping JSON has two keys: "metadata" and "data".
    # Build an index-to-model mapping from the "data" portion.
    data_mapping = mapping_json["data"]
    index2model = build_index2model(data_mapping)
    
    # Retrieve the raw predicted model ID using the index.
    predicted_model_id = index2model.get(pred_model_idx.item(), "Unknown")
    # For the year you can either perform a similar mapping or assume the output is already human‑readable.
    # (Often the year labels are kept as the original strings like "2002".)
    predicted_year = str(pred_year_idx.item())
    
    # Now use the metadata to convert the raw model id into a human-readable name.
    # The metadata section in your JSON was created as:
    #   "metadata": {"make_names": {…}, "model_names": {…}}
    metadata = mapping_json["metadata"]
    # Look up the descriptive model name (if it exists) based on the raw predicted model id.
    readable_model_name = metadata["model_names"].get(predicted_model_id, predicted_model_id)
    
    return readable_model_name, predicted_year

