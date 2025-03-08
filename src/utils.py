import yaml
import pandas as pd

def load_config(config_path="C:/Users/seanf/Desktop/School/Pattern Recognition/CarModelRecognition/config/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def load_class_names_from_csv(csv_path):
    """
    Loads class names from a CSV file that has a 'model_name' column and a 'class' column.
    It returns a list of class names sorted by the 'class' index (0-indexed).
    """
    df = pd.read_csv(csv_path)
    # Ensure we have only unique mappings
    df_unique = df[['class', 'model_name']].drop_duplicates()
    # Sort by the class index so that the order corresponds to model index
    df_sorted = df_unique.sort_values(by='class')
    class_names = df_sorted['model_name'].tolist()
    return class_names