import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class MultiTaskResNet50(nn.Module):
    def __init__(self, num_model_classes, num_year_classes):
        super(MultiTaskResNet50, self).__init__()
        # Load a pretrained ResNet50
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        # Remove the original final fc layer
        backbone.fc = nn.Identity()
        self.backbone = backbone
        
        # Two separate heads for model and year predictions
        self.model_head = nn.Linear(in_features, num_model_classes)
        self.year_head = nn.Linear(in_features, num_year_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        model_out = self.model_head(features)
        year_out = self.year_head(features)
        return model_out, year_out
