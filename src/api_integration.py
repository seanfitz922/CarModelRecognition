import os, json

def build_image_mapping(path):
    mapping = {}
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_dir():
                # For each maker directory, create a new dictionary for models
                maker = entry.name
                maker_path = os.path.join(path, maker)
                mapping[maker] = {}
                with os.scandir(maker_path) as model_entries:
                    for model_entry in model_entries:
                        if model_entry.is_dir():
                            model = model_entry.name
                            model_path = os.path.join(maker_path, model)
                            # Recursively get all image paths (end points) under each model directory
                            mapping[maker][model] = get_all_image_paths(model_path)
    return mapping

def get_all_image_paths(path):
    image_paths = []
    for root, _, files in os.walk(path):
        for file in files:
            # Optionally, filter by image extensions:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    return image_paths

base_path = r"C:\Users\seanf\Desktop\School\Pattern Recognition\CarModelRecognition\data\compcars\data\image"
image_mapping = build_image_mapping(base_path)

# Example of accessing image paths for a specific maker and model:
# print(image_mapping['maker1']['modelA'])

# Define the base directory path
base_path = r"C:\Users\seanf\Desktop\School\Pattern Recognition\CarModelRecognition\data\compcars\data\image"

# Build the nested dictionary representation
directory_tree = build_image_mapping(base_path)

with open('car_tree.json', 'w') as file:
    json.dump(directory_tree, file)
