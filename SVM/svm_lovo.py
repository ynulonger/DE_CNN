import sys
import numpy as np
import scipy.io as sio
from sklearn import svm
import pandas as pd
from sklearn import preprocessing

def get_features(band_index):
	feature_index = np.empty(0)
	for i in band_index:
		band = np.array(range((i-1)*32,i*32))
		feature_index = np.append(feature_index,band)
	feature_index = list(map(int,feature_index))
	return feature_index

def get_vector_deviation(vector1,vector2):
	return vector1-vector2

def get_dataset_deviation(trial_data,base_data):
	new_dataset = np.empty([0,128])
	for i in range(0,2400):
		base_index = i//60
		# print(base_index)
		base_index = 39 if base_index == 40 else base_index
		new_record = get_vector_deviation(trial_data[i],base_data[base_index]).reshape(1,128)
		# print(new_record.shape)
		new_dataset = np.vstack([new_dataset,new_record])
		# print("new shape:",new_dataset.shape)
	return new_dataset

if __name__ == '__main__':
	args = sys.argv[:]
	result = np.empty([0,15])
	for sub_id in range(1,33):
		print("processing ",sub_id)
		arousal_or_valence = args[1]
		sub_id = "%02d" % sub_id
		sub = "s"+str(sub_id)
		file = sio.loadmat("/home/yyl/DE_CNN/1D_dataset/DE_"+sub+".mat")

		X = file["data"]
		base_data = file["base_data"]

		X = get_dataset_deviation(X,base_data)
		X = preprocessing.scale(X, axis=1, with_mean=True,with_std=True,copy=True)
		y = np.squeeze(file[arousal_or_valence+"_labels"].transpose())
		# shuffle data
		# index = np.array(range(0,len(y)))
		# np.random.shuffle(index)
		# input_X = X[index]
		# y = y[index]

		fold = 40
		acc_list = np.empty(0)

		mean_accuracy = 0

		for curr_fold in range(fold):
			fold_size = X.shape[0]//fold
			indexes_list = [i for i in range(len(X))]
			indexes = np.array(indexes_list)
			split_list = [i for i in range(curr_fold*fold_size,(curr_fold+1)*fold_size)]
			split = np.array(split_list)

			test_x = X[split] 
			test_y = y[split]

			split = np.array(list(set(indexes_list)^set(split_list)))
			train_x = X[split]
			train_y = y[split]

			# print("train_x shape:",train_x.shape)
			# print("test_x shape:",test_x.shape)
			
			train_sample = train_y.shape[0]
			test_sample = test_y.shape[0]
			
			clf = svm.SVC()
			clf.fit(train_x,train_y)

			Z = clf.predict(test_x)
			accuracy = np.sum(Z==test_y)/len(Z)
			# print("fold:",curr_fold,"acc:",accuracy)
			mean_accuracy += accuracy

			# print(count,"---",key,mean_accuracy/fold*100)
		acc_list = np.append(acc_list,mean_accuracy/fold*100)

		print(acc_list)
	# print(result)
	accuracy = pd.DataFrame(acc_list)
	accuracy.columns = [arousal_or_valence]
	writer = pd.ExcelWriter("/home/yyl/DE_CNN/SVM/result/lovo_"+arousal_or_valence+".xlsx")
	accuracy.to_excel(writer, 'result', index=False)
	writer.save()

