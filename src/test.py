import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import CarModelCNN  # Your model definition
from utils import load_config, load_class_names_from_csv  # Adjust as needed
import pandas as pd


# Step 1: Load configuration (assumes config contains keys: model_save_path, num_classes, etc.)
config = load_config()
NUM_CLASSES = config['num_classes']
model_path = config['model_save_path'] + "/best_model.pth"  # or "best_model.pth"

# Step 2: Instantiate the model and load state dict
model = CarModelCNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # set model to evaluation mode

# Step 3: Define the same transforms as used during training
inference_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Step 4: Load class names from the CSV
class_names = load_class_names_from_csv(config['car_names_csv'])  # Returns a list of 196 class names

# Step 5: Create a helper function to predict a single image
def predict_image(image_path, model, transform, class_names=None):
    # Load image and apply transform
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # add batch dimension
    
    # Get prediction from the model
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = probabilities.argmax(dim=1).item()
    
    if class_names:
        predicted_class = class_names[predicted_idx]
    else:
        predicted_class = predicted_idx
        
    return predicted_class, probabilities.squeeze().tolist()

# Example usage:
image_path = "C:/Users/seanf/Desktop/School/Pattern Recognition/CarModelRecognition/data/cars_test/cars_test/00066.jpg"
predicted_class, probs = predict_image(image_path, model, inference_transforms, class_names)
print("Predicted class:", predicted_class)

# Optionally, display the image and its prediction:
img = Image.open(image_path).convert('RGB')
plt.imshow(img)
plt.title(f"Predicted: {predicted_class}")
plt.axis("off")
plt.show()
