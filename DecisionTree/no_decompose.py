import os
import sys
import math
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
from scipy.signal import butter, lfilter

def read_file(file):
	data = sio.loadmat(file)
	data = data['data']
	# print(data.shape)
	return data

def compute_DE(signal):
	variance = np.var(signal,ddof=1)
	return math.log(2*math.pi*math.e*variance)/2

def compute_dataset_DE(file):
	# trial*channel*sample
	start_index = 384 #3s pre-trial signals
	data = read_file(file)
	shape = data.shape
	frequency = 128

	decomposed_de = np.empty([0,1,60])

	base_DE = np.empty([0,32])

	for trial in range(40):
		temp_base_DE = np.empty([0])
		temp_base___DE = np.empty([0])

		temp_de = np.empty([0,60])

		for channel in range(32):
			trial_signal = data[trial,channel,384:]
			base_signal = data[trial,channel,:384]
			#****************compute base DE****************

			base___DE =(compute_DE(base_signal[:128])+compute_DE(base_signal[128:256])+compute_DE(base_signal[256:]))/3

			temp_base___DE = np.append(temp_base___DE,base___DE)

			DE__ = np.zeros(shape=[0],dtype = float)

			for index in range(60):
				DE__ =np.append(DE__,compute_DE(base_signal[index*frequency:(index+1)*frequency]))

			temp_de = np.vstack([temp_de,DE__])
		temp_trial_de = temp_de.reshape(-1,1,60)
		decomposed_de = np.vstack([decomposed_de,temp_trial_de])

		temp_base_DE = np.append(temp_base_DE,temp_base___DE)
		base_DE = np.vstack([base_DE,temp_base_DE])
	decomposed_de = decomposed_de.reshape(-1,32,1,60).transpose([0,3,2,1]).reshape(-1,1,32).reshape(-1,32)
	print("base_DE shape:",base_DE.shape)
	print("trial_DE shape:",decomposed_de.shape)
	return base_DE,decomposed_de

def get_labels(file):
	#0 valence, 1 arousal, 2 dominance, 3 liking
	valence_labels = sio.loadmat(file)["labels"][:,0]>5	# valence labels
	arousal_labels = sio.loadmat(file)["labels"][:,1]>5	# arousal labels
	final_valence_labels = np.empty([0])
	final_arousal_labels = np.empty([0])
	for i in range(len(valence_labels)):
		for j in range(0,60):
			final_valence_labels = np.append(final_valence_labels,valence_labels[i])
			final_arousal_labels = np.append(final_arousal_labels,arousal_labels[i])
	print("labels:",final_arousal_labels.shape)
	return final_arousal_labels,final_valence_labels

if __name__ == '__main__':
	dataset_dir = "/home/data_preprocessed_matlab/"
	
	result_dir = "/home/yyl/DE_CNN/1D_dataset/full/"

	for file in os.listdir(dataset_dir):
		print("processing: ",file,"......")
		file_path = os.path.join(dataset_dir,file)
		base_DE,trial_DE = compute_dataset_DE(file_path)
		arousal_labels,valence_labels = get_labels(file_path)
		sio.savemat(result_dir+file,{"base_data":base_DE,"data":trial_DE,"valence_labels":valence_labels,"arousal_labels":arousal_labels})
