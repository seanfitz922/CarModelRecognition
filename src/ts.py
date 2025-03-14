import matplotlib.pyplot as plt
import pandas as pd
from utils import load_config
import os
from PIL import Image

config = load_config()

# Load the CSV file
train_df = pd.read_csv(config['train_csv_path'])

# Sample 20 random rows (using the same sample for both image paths and titles)
sample_data = train_df.sample(20)

# If your CSV has a full image path column, use that; otherwise, use the "image" column
if "image_path" in sample_data.columns:
    sample_images = sample_data["image_path"].tolist()
else:
    # Use a base directory if only file names are provided in the CSV
    base_path = "C:/Users/seanf/Desktop/School/Pattern Recognition/CarModelRecognition/data/"
    sample_images = sample_data["image"].tolist()
    # Prepend the base_path to each filename
    sample_images = [os.path.join(base_path, img_name) for img_name in sample_images]

# Use the "true_class_name" column for the titles
sample_titles = sample_data["true_class_name"].tolist()

# Create a figure to display the images
plt.figure(figsize=(15, 10))

# Loop through the images and plot them
for i, (img_path, title) in enumerate(zip(sample_images, sample_titles)):
    try:
        # If the image path is not absolute, join it with the base path
        if not os.path.isabs(img_path):
            base_path = "C:/Users/seanf/Desktop/School/Pattern Recognition/CarModelRecognition/data/"
            img_path = os.path.join(base_path, img_path)
        
        # Open and display the image
        image = Image.open(img_path)
        plt.subplot(4, 5, i + 1)  # 4 rows, 5 columns, index starts at 1
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")

plt.tight_layout()
plt.show()
