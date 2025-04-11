import yaml
import pandas as pd
import random
import json
import torch
import timm
import pandas as pd


# load yaml with hyperparameters
def load_config(config_path="C:/Users/seanf/Desktop/School/Pattern Recognition/CarModelRecognition/config/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

# load all class names (deprecated)
def load_class_names_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    # only unique mappings
    df_unique = df[['class', 'model_name']].drop_duplicates()
    df_sorted = df_unique.sort_values(by='class')
    class_names = df_sorted['model_name'].tolist()
    return class_names

# from https://www.geeksforgeeks.org/convert-json-to-dictionary-in-python/
# load image json file into dictionary
def load_json_dict(json_dict_path):
    with open(json_dict_path) as json_file:
        json_data = json.load(json_file)

    return json_data

# iterate through dict and return image path
def parse_image_dict(data):
    lookup = {}
    for make_id, make_data in data.items():
        for model_id, model_data in make_data.items():
            for year, images in model_data.items():
                for image_info in images:
                    label = image_info.get("label", {})
                    # Build a combined record with all desired fields.
                    combined_record = {
                        "image_path": image_info['image_path'],
                        "model_id": model_id,
                        "year": year,
                        "viewpoint": label.get("viewpoint"),
                        "bbox": label.get("bbox")
                    }
                    rel_path = f"{make_id}/{model_id}/{year}/{image_info['image_id']}.jpg"
                    lookup[rel_path] = combined_record
    return lookup


# from https://stackoverflow.com/questions/3925614/how-do-you-read-a-file-into-a-list-in-python
def load_file(path):
    with open(path, 'r') as file:
        return file.read().splitlines()
    
def build_records(json_dict_path, file_txt):
    json_mapping = load_json_dict(json_dict_path)
    lookup_dict = parse_image_dict(json_mapping)
    train_lines = load_file(file_txt)

    results = []
    for relative_path in train_lines:
        record = lookup_dict.get(relative_path)
        if record is None:
            print(f"Path not found: {relative_path}")
            continue

        results.append({
            "image_path": record["image_path"],
            "model_id": record.get("model_id"),
            "year": record.get("year")
            #"viewpoint": record.get("viewpoint"),  # If you later want to use these
            #"bbox": record.get("bbox")
        })
    
    return results

# (path, model_id, year, viewpoint, bbox)
# ('C:/Users/seanf/Desktop/School/Pattern Recognition/CarModelRecognition/data/compcars/data/image/78/1/2010/439374a1456969.jpg', '1', '2010', 1, [131, 111, 765, 568])

# training split 
def random_split(train_data):
    num_samples = len(train_data)
    train_ratio = 0.8
    train_size = int(train_ratio * num_samples)
    indices = list(range(num_samples))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    return train_indices, val_indices

def load_model():
    config = load_config()
    model = timm.create_model('efficientnetv2_rw_m', pretrained=False, num_classes=config['num_classes'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(config['final_model_path'], map_location=device))
    model.to(device)
    model.eval()
    return model, device