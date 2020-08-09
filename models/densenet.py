import torch
import torch.nn as nn
import torchvision.models as models

class DenseNet(nn.Module):
    def __init__(self, dataset, pretrained=True):
        super(DenseNet, self).__init__()
        num_classes = 50 if dataset=="ESC" else 10
        self.model = models.densenet201(pretrained=pretrained)
        self.model.classifier = nn.Linear(1920, num_classes)
        
    def forward(self, x):
        output = self.model(x)
        return output
