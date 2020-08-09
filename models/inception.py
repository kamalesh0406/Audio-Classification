import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

class Inception(nn.Module):
	def __init__(self, dataset, pretrained=True):
		super(Inception, self).__init__()
		num_classes = 50 if dataset=="ESC" else 10
		self.model = models.inception_v3(pretrained=pretrained, aux_logits=False)
		self.model.fc = nn.Linear(2048, num_classes)

	def forward(self, x):
		output = self.model(x)
		return output
