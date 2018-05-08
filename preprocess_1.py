import time
from sklearn import preprocessing
from scipy.signal import butter, lfilter
import scipy.io as sio
import numpy as np
import os
import math
import sys
# import matplotlib.pyplot as plt

def read_file(file):
	data = sio.loadmat(file)
	data = data['data']
	print(data.shape)
	return data

def compute_DE(signal):
	variance = np.var(signal,ddof=1)
	return math.log(2*math.pi*math.e*variance)/2

def decompose(file,use_baseline):
	# trial*channel*sample
	start_index = 384 #3s pre-trial signals
	data = read_file(file)
	shape = data.shape
	frequency = 128
	decomposed_data = np.empty([0,1,8064])
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

			DE= np.zeros(shape=[0],dtype = float)

			for index in range(60):
				DE =np.append(DE,compute_DE(trial_signal[index*frequency:(index+1)*frequency])-base_DE)
			temp_de = np.vstack([temp_de,DE])
			# print("trial:",trial,",channel:",channel+1,temp_data.shape)
		temp_trial_de = temp_de.reshape(-1,1,60)
		# print("trial:",trial, temp_trial.shape)
		# print("trial_de:",trial, temp_trial_de.shape)
		decomposed_de = np.vstack([decomposed_de,temp_trial_de])
	decomposed_de = decomposed_de.reshape(-1,32,1,60)
	return decomposed_de

def data_1Dto2D(data, Y=9, X=9):
	data_2D = np.zeros([Y, X])
	data_2D[0] = (0,        0,          0,          data[0],    0,          data[16],   0,          0,          0       )
	data_2D[1] = (0,        0,          0,          data[1],    0,          data[17],   0,          0,          0       )
	data_2D[2] = (data[3],  0,          data[2],    0,          data[18],   0,          data[19],   0,          data[20])
	data_2D[3] = (0,        data[4],    0,          data[5],    0,          data[22],   0,          data[21],   0       )
	data_2D[4] = (data[7],  0,          data[6],    0,          data[23],   0,          data[24],   0,          data[25])
	data_2D[5] = (0,        data[8],    0,          data[9],    0,          data[27],   0,          data[26],   0       )
	data_2D[6] = (data[11], 0,          data[10],   0,          data[15],   0,          data[28],   0,          data[29])
	data_2D[7] = (0,        0,          0,          data[12],   0,          data[30],   0,          0,          0       )
	data_2D[8] = (0,        0,          0,          data[13],   data[14],   data[31],   0,          0,          0       )
	# return shape:9*9
	return data_2D

def feature_normalize(data):
	mean = data[data.nonzero()].mean()
	sigma = data[data. nonzero ()].std()
	data_normalized = data
	data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
	return data_normalized

def pre_process(path,y_n):
	decomposed_de = decompose(path,y_n)
	#0 valence, 1 arousal, 2 dominance, 3 liking
	valence_labels = sio.loadmat(path)["labels"][:,0]>5	# valence labels
	arousal_labels = sio.loadmat(path)["labels"][:,1]>5	# arousal labels
	
	final_valence_labels = np.empty([0])
	final_arousal_labels = np.empty([0])
	for i in range(len(valence_labels)):
		for j in range(0,60):
			final_valence_labels = np.append(final_valence_labels,valence_labels[i])
			final_arousal_labels = np.append(final_arousal_labels,arousal_labels[i])
	data_inter_cnn = np.empty([0,9,9])

	decomposed_de = decomposed_de.transpose([0,3,2,1])
	decomposed_de = decomposed_de.reshape(-1,1,32)	# 2400*4*32
	samples = decomposed_de.shape[0]
	bands = decomposed_de.shape[1]
	data_cnn = np.empty([0,9,9])
	for sample in range(samples):
		for band in range(bands):
			data_2D_temp = feature_normalize(data_1Dto2D(decomposed_de[sample,band,:]))
			data_2D_temp = data_2D_temp.reshape(1,9,9)
			# print("data_2d_temp shape:",data_2D_temp.shape)
			data_cnn = np.vstack([data_cnn,data_2D_temp])
	data_cnn = data_cnn.reshape(-1,1,9,9)
	# print("final data shape:",data_cnn.shape)
	return data_cnn,final_valence_labels,final_arousal_labels

if __name__ == '__main__':
	dataset_dir = "/home/data_preprocessed_matlab/"
	use_baseline = sys.argv[1]
	if use_baseline=="T":
		result_dir = "/home/yyl/DE_CNN/DE_dataset/without_decomposed/DE_"
	else:
		result_dir = "/home/yyl/DE_CNN/DE_dataset/without_decomposed/without_base/DE_"
	for file in os.listdir(dataset_dir):
		print("processing: ",file,"......")
		file_path = os.path.join(dataset_dir,file)
		data,valence_labels,arousal_labels = pre_process(file_path,use_baseline)
		print("final shape:",data.shape)
		sio.savemat(result_dir+file,{"data":data,"valence_labels":valence_labels,"arousal_labels":arousal_labels})
		