from scipy.io import loadmat
import json
from utils import load_config



config = load_config()

def load_mapping_from_json():
    json_path = config['json_dict_path']
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    unique_models = set()
    unique_years = set()
    
    # Traverse the nested dictionary:
    # data structure: {maker: {model: {year: [list of records]}}}
    for maker, models in data.items():
        for model, years in models.items():
            unique_models.add(model)
            for year in years.keys():
                unique_years.add(year)
    
    # Sort the unique values. (The same sorting logic as in training is critical!)
    sorted_models = sorted(list(unique_models), key=lambda x: int(x))
    sorted_years = sorted(list(unique_years), key=lambda x: (x == "unknown", x))
    
    # Build mappings:
    model2index = {model: idx for idx, model in enumerate(sorted_models)}
    year2index  = {year: idx for idx, year in enumerate(sorted_years)}
    
    # Invert the mappings so you can convert from predicted index to original label
    index2model = {v: k for k, v in model2index.items()}
    index2year  = {v: k for k, v in year2index.items()}
    
    return index2model, index2year

def load_model_name_mapping():
    mat_file_path = config['make_model_name_mat']
    data = loadmat(mat_file_path)
    model_names_cell = data.get("model_names")
    
    if model_names_cell is None:
        raise ValueError("Key 'model_names' not found in the .mat file.")
    
    mapping = {}
    for i, element in enumerate(model_names_cell.flatten()):
        # Check if the element has at least one entry.
        if element.size > 0:
            # Extract the string
            mapping[str(i+1)] = element[0]
        else:
            mapping[str(i+1)] = "Unknown"
    
    return mapping

