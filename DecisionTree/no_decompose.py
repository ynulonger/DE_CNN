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

def compute_dataset_DE(file,use_baseline):
	# trial*channel*sample
	start_index = 384 #3s pre-trial signals
	data = read_file(file)
	shape = data.shape
	frequency = 128
	decomposed_de = np.empty([0,1,60])
	for trial in range(40):
		temp_data = np.empty([0,8064])
		temp_de = np.empty([0,60])
		for channel in range(32):
			trial_signal = data[trial,channel,384:]
			if use_baseline =='T':
				base_signal = data[trial,channel,:384]
				#****************compute base DE****************
				base_DE = (compute_DE(base_signal[:128])+compute_DE(base_signal[128:256])+compute_DE(base_signal[256:]))/3
			else:
				base_DE=0

			DE = np.zeros(shape=[0],dtype = float)

			for index in range(60):
				DE =np.append(DE,compute_DE(trial_signal[index*frequency:(index+1)*frequency])-base_DE)

			temp_de = np.vstack([temp_de,DE])
			# print("trial:",trial,",channel:",channel+1,temp_data.shape)
		temp_trial_de = temp_de.reshape(-1,1,60)

		decomposed_de = np.vstack([decomposed_de,temp_trial_de])
	decomposed_de = decomposed_de.reshape(-1,32,1,60)
	return decomposed_de

def feature_normalize(data):
	mean = data[data.nonzero()].mean()
	sigma = data[data. nonzero ()].std()
	data_normalized = data
	data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
	return data_normalized

def pre_process(path,y_n):
	decomposed_de = compute_dataset_DE(path,y_n)
	#0 valence, 1 arousal, 2 dominance, 3 liking
	valence_labels = sio.loadmat(path)["labels"][:,0]>5	# valence labels
	arousal_labels = sio.loadmat(path)["labels"][:,1]>5	# arousal labels
	
	final_valence_labels = np.empty([0])
	final_arousal_labels = np.empty([0])
	for i in range(len(valence_labels)):
		for j in range(0,60):
			final_valence_labels = np.append(final_valence_labels,valence_labels[i])
			final_arousal_labels = np.append(final_arousal_labels,arousal_labels[i])

	decomposed_de = decomposed_de.transpose([0,3,2,1])
	decomposed_de = decomposed_de.reshape(-1,1,32)	# 2400*1*32
	samples = decomposed_de.shape[0]
	bands = decomposed_de.shape[1]
	data_cnn = np.empty([0,128])
	for sample in range(samples):
		temp_data_cnn = np.empty([0,128])
		for band in range(bands):
			decomposed_de[sample,band,:] = feature_normalize(decomposed_de[sample,band,:])
	# print("final data shape:",data_cnn.shape)
	decomposed_de = decomposed_de.transpose([0,2,1])
	decomposed_de = decomposed_de.reshape(-1,32)
	return decomposed_de,final_valence_labels,final_arousal_labels

if __name__ == '__main__':
	dataset_dir = "/home/data_preprocessed_matlab/"
	use_baseline = sys.argv[1]
	if use_baseline=="T":
		result_dir = "/home/yyl/DE_CNN/DecisionTree/with_base/without_decomposed/DE_"
	else:
		result_dir = "/home/yyl/DE_CNN/DecisionTree/without_base/without_decomposed/DE_"
	for file in os.listdir(dataset_dir):
		print("processing: ",file,"......")
		file_path = os.path.join(dataset_dir,file)
		data,valence_labels,arousal_labels = pre_process(file_path,use_baseline)
		print("final shape:",data.shape)
		sio.savemat(result_dir+file,{"data":data,"valence_labels":valence_labels,"arousal_labels":arousal_labels})
