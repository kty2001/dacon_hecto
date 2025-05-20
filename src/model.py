from torchvision import models


def create_model(model: str = 'resnet'):
    if model == 'resnet':
        return _resnet18_pretrained()
    elif model == 'efficientnet':
        return _efficientb0_pretrained()

def _resnet18_pretrained():
    return models.resnet18(weights=(models.ResNet18_Weights.IMAGENET1K_V1))

def _efficientb0_pretrained():
    return models.efficientnet_b0(weights=(models.EfficientNet_B0_Weights.IMAGENET1K_V1))