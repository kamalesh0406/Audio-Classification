import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
	def __init__(self, dataset, pretrained=True):
		super(ResNet, self).__init__()
		num_classes = 50 if dataset=="ESC" else 10
		self.model = models.resnet50(pretrained=pretrained)
		self.model.fc = nn.Linear(2048, num_classes)
		
	def forward(self, x):
		output = self.model(x)
		return output