import os
import json
import pprint

def parse_label_file(label_file):
    """
    Reads a label file and returns its content in a dictionary.
    Expects three lines:
      1. Viewpoint annotation (integer: -1, 1, 2, 3, 4, or 5)
      2. Number of bounding boxes (currently always 1)
      3. Bounding box coordinates in the format 'x1 y1 x2 y2'
    """
    with open(label_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    if len(lines) < 3:
        raise ValueError(f"Label file {label_file} doesn't have the required three lines.")
    
    viewpoint = int(lines[0])
    num_bboxes = int(lines[1])
    bbox_coords = list(map(int, lines[2].split()))
    
    if len(bbox_coords) != 4:
        raise ValueError(f"Label file {label_file} does not have four coordinates.")
    
    return {
        "viewpoint": viewpoint,
        "num_bboxes": num_bboxes,
        "bbox": bbox_coords
    }

def build_image_mapping(image_base_path):
    """
    Build a nested dictionary mapping for images with the structure:
      image_mapping[maker][model][year][image_id] = {"image_path": ...}
    """
    mapping = {}
    for maker in os.listdir(image_base_path):
        maker_path = os.path.join(image_base_path, maker)
        if not os.path.isdir(maker_path):
            continue
        mapping[maker] = {}
        for model in os.listdir(maker_path):
            model_path = os.path.join(maker_path, model)
            if not os.path.isdir(model_path):
                continue
            mapping[maker][model] = {}
            for year in os.listdir(model_path):
                year_path = os.path.join(model_path, year)
                if not os.path.isdir(year_path):
                    continue
                mapping[maker][model][year] = {}
                for file in os.listdir(year_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                        # Use the filename without extension as the image ID
                        image_id = os.path.splitext(file)[0]
                        image_path = os.path.join(year_path, file)
                        mapping[maker][model][year][image_id] = {"image_path": image_path, "label": None}
    return mapping

def build_label_mapping(label_base_path):
    """
    Build a nested dictionary mapping for labels with the structure:
      label_mapping[maker][model][year][image_id] = label_data
    """
    mapping = {}
    for maker in os.listdir(label_base_path):
        maker_path = os.path.join(label_base_path, maker)
        if not os.path.isdir(maker_path):
            continue
        mapping[maker] = {}
        for model in os.listdir(maker_path):
            model_path = os.path.join(maker_path, model)
            if not os.path.isdir(model_path):
                continue
            mapping[maker][model] = {}
            for year in os.listdir(model_path):
                year_path = os.path.join(model_path, year)
                if not os.path.isdir(year_path):
                    continue
                mapping[maker][model][year] = {}
                for file in os.listdir(year_path):
                    if file.lower().endswith('.txt'):
                        image_id = os.path.splitext(file)[0]
                        label_file = os.path.join(year_path, file)
                        label_data = parse_label_file(label_file)
                        mapping[maker][model][year][image_id] = label_data
    return mapping

def combine_mappings(image_mapping, label_mapping):
    """
    Combine the image and label mappings into a single nested dictionary.
    The combined mapping has the structure:
      combined[maker][model][year] = list of { "image_id", "image_path", "label" }
    """
    combined = {}
    for maker, maker_data in image_mapping.items():
        combined[maker] = {}
        for model, model_data in maker_data.items():
            combined[maker][model] = {}
            for year, year_data in model_data.items():
                combined[maker][model][year] = []
                for image_id, image_entry in year_data.items():
                    # Get the label for the current image, if it exists
                    label = (label_mapping.get(maker, {})
                                        .get(model, {})
                                        .get(year, {})
                                        .get(image_id))
                    combined[maker][model][year].append({
                        "image_id": image_id,
                        "image_path": image_entry["image_path"],
                        "label": label
                    })
    return combined

def save_mapping_to_json(mapping, filename):
    """Save the mapping dictionary to a JSON file."""
    with open(filename, "w") as f:
        json.dump(mapping, f, indent=2)

def main():
    # Set your base paths for images and labels.
    image_base_path = r"C:\Users\seanf\Desktop\School\Pattern Recognition\CarModelRecognition\data\compcars\data\image"
    label_base_path = r"C:\Users\seanf\Desktop\School\Pattern Recognition\CarModelRecognition\data\compcars\data\label"
    
    print("Building image mapping...")
    image_mapping = build_image_mapping(image_base_path)
    
    print("Building label mapping...")
    label_mapping = build_label_mapping(label_base_path)
    
    print("Combining mappings...")
    combined_mapping = combine_mappings(image_mapping, label_mapping)
    
    # Pretty-print the combined mapping for inspection
    pprint.pprint(combined_mapping)
    
    # Save the combined mapping as a JSON file
    output_json_file = "combined_mapping.json"
    save_mapping_to_json(combined_mapping, output_json_file)
    print(f"Combined mapping saved to {output_json_file}")

if __name__ == "__main__":
    main()
