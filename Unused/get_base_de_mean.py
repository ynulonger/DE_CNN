import time
from sklearn import preprocessing
from scipy.signal import butter, lfilter
import scipy.io as sio
import numpy as np
import os
import math
import sys
import pandas as pd

def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y

def read_file(file):
	data = sio.loadmat(file)
	data = data['data']
	print(data.shape)
	return data

def compute_DE(signal):
	variance = np.var(signal,ddof=1)
	return math.log(2*math.pi*math.e*variance)/2

def decompose(file):
	# trial*channel*sample
	start_index = 384 #3s pre-trial signals
	data = read_file(file)
	shape = data.shape
	frequency = 128
	decomposed_data = np.empty([0,4,8064])
	decomposed_de = np.empty([0,4,3])
	for trial in range(40):
		temp_data = np.empty([0,8064])
		temp_de = np.empty([0,3])
		for channel in range(32):
			trial_signal = data[trial,channel,384:]
			base_signal = data[trial,channel,:384]
			#****************compute base DE****************
			base_theta = butter_bandpass_filter(base_signal, 4, 8, frequency, order=3)
			base_alpha = butter_bandpass_filter(base_signal, 8,14, frequency, order=3)
			base_beta = butter_bandpass_filter(base_signal,14,31, frequency, order=3)
			base_gmma = butter_bandpass_filter(base_signal,31,45, frequency, order=3)

			base_theta_DE = (compute_DE(base_theta[:128])+compute_DE(base_theta[128:256])+compute_DE(base_theta[256:]))/3
			base_alpha_DE =(compute_DE(base_alpha[:128])+compute_DE(base_alpha[128:256])+compute_DE(base_alpha[256:]))/3
			base_beta_DE =(compute_DE(base_beta[:128])+compute_DE(base_beta[128:256])+compute_DE(base_beta[256:]))/3
			base_gmma_DE =(compute_DE(base_gmma[:128])+compute_DE(base_gmma[128:256])+compute_DE(base_gmma[256:]))/3


			theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
			alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
			beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
			gmma = butter_bandpass_filter(trial_signal, 31, 45, frequency, order=3)

			DE_theta = DE_alpha= DE_beta = DE_gmma= np.zeros(shape=[0],dtype = float)

			for index in range(3):
				DE_theta =np.append(DE_theta,compute_DE(base_theta[index*frequency:(index+1)*frequency]))
				DE_alpha =np.append(DE_alpha,compute_DE(base_alpha[index*frequency:(index+1)*frequency]))
				DE_beta =np.append(DE_beta,compute_DE(base_beta[index*frequency:(index+1)*frequency]))
				DE_gmma =np.append(DE_gmma,compute_DE(base_gmma[index*frequency:(index+1)*frequency]))
			temp_de = np.vstack([temp_de,DE_theta])
			temp_de = np.vstack([temp_de,DE_alpha])
			temp_de = np.vstack([temp_de,DE_beta])
			temp_de = np.vstack([temp_de,DE_gmma])
			# print("trial:",trial,",channel:",channel+1,temp_data.shape)
		# temp_trial = temp_data.reshape(-1,4,8064)
		temp_trial_de = temp_de.reshape(-1,4,3)
		# print("trial:",trial, temp_trial.shape)
		# print("trial_de:",trial, temp_trial_de.shape)
		# decomposed_data = np.vstack([decomposed_data,temp_trial])
		decomposed_de = np.vstack([decomposed_de,temp_trial_de])
	# decomposed_data = decomposed_data.reshape(-1,32,4,8064)
	decomposed_de = decomposed_de.reshape(-1,32,4,3)
	return decomposed_de

DE = decompose("/home/data_preprocessed_matlab/s01.mat")
print(DE.shape)
DE= np.transpose(DE,[0,1,2,3])
DE = DE.reshape(-1,128)
print(DE.shape)
dictionary = {str(i):DE[:,i] for i in range(0,128)}
result = pd.DataFrame(dictionary)
writer = pd.ExcelWriter("/home/yyl/DE_CNN/test.xlsx")
result.to_excel(writer, 'result', index=False)
writer.save()

