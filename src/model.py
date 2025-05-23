import torch
from torchvision import models


def create_model(model: str = 'resnet', num_classes: int = 396):
    if model == 'resnet':
        return _resnet18_pretrained(num_classes)
    elif model == 'efficientnet':
        return _efficientb0_pretrained(num_classes)

def _resnet18_pretrained(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def _efficientb0_pretrained(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    return model