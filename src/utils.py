import yaml
import pandas as pd
import random

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

# training split 
def random_split(train_df):
    num_samples = len(train_df)
    train_ratio = 0.8
    train_size = int(train_ratio * num_samples)
    indices = list(range(num_samples))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    return train_indices, val_indices