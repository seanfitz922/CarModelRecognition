import torch
import torch.nn as nn
import torch.nn.functional as F

class CarModelCNN(nn.Module):
    def __init__(self, num_classes):
        super(CarModelCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2,2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # Ensure fixed feature map size

        # Dynamically determine input size for fc1
        self.fc1 = nn.Linear(self._get_fc_input_dim(), 1024)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, num_classes)

    def _get_fc_input_dim(self):
        with torch.no_grad():
            sample_input = torch.zeros(1, 3, 224, 224)  # Assuming input images are 224x224
            sample_output = self._forward_features(sample_input)
            return sample_output.view(1, -1).shape[1]

    def _forward_features(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.adaptive_pool(x)  # Ensures the feature map size is always (batch, 512, 4, 4)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        #print(f"Shape before flattening: {x.shape}") 
        x = x.view(x.size(0), -1)  # Flatten
        #print(f"Shape after flattening: {x.shape}")  
        
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)

        return x
