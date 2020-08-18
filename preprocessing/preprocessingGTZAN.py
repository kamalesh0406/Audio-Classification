import librosa
import argparse
import pandas as pd
import numpy as np
import pickle as pkl 
import torch
import torchaudio
import torchvision
from PIL import Image
import os
from joblib import dump

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--store_dir", type=str)
parser.add_argument("--sampling_rate", default=22050, type=int)

def extract_spectrogram(values, clip, target):
	num_channels = 3
	window_sizes = [25, 50, 100]
	hop_sizes = [10, 25, 50]

	specs = []

	for i in range(num_channels):
		window_length = int(round(window_sizes[i]*args.sampling_rate/1000))
		hop_length = int(round(hop_sizes[i]*args.sampling_rate/1000))

		clip = torch.Tensor(clip)
		spec = torchaudio.transforms.MelSpectrogram(sample_rate=args.sampling_rate, n_fft=2205, win_length=window_length, hop_length=hop_length, n_mels=128)(clip) #Check this otherwise use 2400
		eps = 1e-6
		spec = spec.numpy()
		spec = np.log(spec+ eps)
		spec = np.asarray(torchvision.transforms.Resize((128, 1500))(Image.fromarray(spec)))
		specs.append(spec)

	new_entry = {}
	new_entry["audio"] = clip.numpy()
	new_entry["values"] = np.array(specs)
	new_entry["target"] = target
	values.append(new_entry)
def extract_features(audios):
	values = []
	for audio in audios:
		try:
			clip, sr = librosa.load(audio["name"], sr=args.sampling_rate)
		except:
			continue
		extract_spectrogram(values, clip, audio["class_idx"])
		print("Finished audio {}".format(audio))
	return values
if __name__=="__main__":
	args = parser.parse_args()
	root_dir = args.data_dir

	training_audios = []
	validation_audios = []

	for root, dirs, files in os.walk(root_dir):
		class_names = dirs
		break

	for _class in class_names:
		class_dir = os.path.join(root_dir, _class)
		class_audio = []
		for root, dirs, files in os.walk(class_dir):
			for file in files:
				if file.endswith('.wav'):
					class_audio.append({"name":os.path.join(root, file), "class_idx": class_names.index(_class)})

		training_audios.extend(class_audio[:int(len(class_audio)*4/5)])
		validation_audios.extend(class_audio[int(len(class_audio)*4/5):])

	training_values = extract_features(training_audios)
	with open("{}training128mel1.pkl".format(args.store_dir),"wb") as handler:
		pkl.dump(training_values, handler, protocol=pkl.HIGHEST_PROTOCOL)

	validation_values = extract_features(validation_audios)
	with open("{}validation128mel1.pkl".format(args.store_dir),"wb") as handler:
		pkl.dump(validation_values, handler, protocol=pkl.HIGHEST_PROTOCOL)