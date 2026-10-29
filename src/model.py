import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    # Transfer learning: congela o backbone e treina só a última camada
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model