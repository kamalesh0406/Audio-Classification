from torch.utils.data import *
import lmdb
import torchvision
import pandas as pd
import numpy as np
import pickle
import torch
from PIL import Image

class AudioDataset(Dataset):
	def __init__(self, pkl_dir, dataset_name, transforms=None):
		self.data = []
		self.length = 1500 if dataset_name=="GTZAN" else 250
		self.transforms = transforms
		with open(pkl_dir, "rb") as f:
			self.data = pickle.load(f)
	def __len__(self):
		return len(self.data)
	def __getitem__(self, idx):
		entry = self.data[idx]
		output_data = {}
		values = entry["values"].reshape(-1, 128, self.length)
		values = torch.Tensor(values)
		if self.transforms:
			values = self.transforms(values)
		target = torch.LongTensor([entry["target"]])
		return (values, target)

def fetch_dataloader(pkl_dir, dataset_name, batch_size, num_workers):
	dataset = AudioDataset(pkl_dir, dataset_name)
	dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
	return dataloader