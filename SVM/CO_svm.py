import sys
import numpy as np
import scipy.io as sio
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

def get_features(band_index):
	feature_index = np.empty(0)
	for i in band_index:
		band = np.array(range((i-1)*32,i*32))
		feature_index = np.append(feature_index,band)
	feature_index = list(map(int,feature_index))
	return feature_index

if __name__ == '__main__':
	args = sys.argv[:]
	arousal_or_valence = args[1]
	acc_list2 = np.empty(0)
	acc_list1 = np.empty(0)
	all_data = np.empty([0,2400,128])
	all_valence_labels = np.empty([0])
	for sub_id in range(1,33):
		print("processing ",sub_id)
		
		sub_id = "%02d" % sub_id
		sub = "s"+str(sub_id)
		file = sio.loadmat("/home/yyl/DE_CNN/DecisionTree/with_base/without_decomposed/DE_"+sub+".mat")
		X = file["data"]
		y = np.squeeze(file[arousal_or_valence+"_labels"].transpose())
		# shuffle data
		index = np.array(range(0,len(y)))
		np.random.shuffle(index)
		input_X = X[index]
		y = y[index]

		fold = 10

		count = 0
		
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
			train_sample = train_y.shape[0]
			test_sample = test_y.shape[0]
			# 训练模型，限制树的最大深度
			clf = DecisionTreeClassifier(max_depth=20)
			clf.fit(train_x,train_y)

			Z = clf.predict(test_x)
			accuracy = np.sum(Z==test_y)/len(Z)
			# print("fold:",curr_fold,"acc:",accuracy)
			mean_accuracy += accuracy

		# print(count,"---",key,mean_accuracy/fold*100)
		count+=1
		acc_list1 = np.append(acc_list1,mean_accuracy/fold*100)
	print(acc_list1)

	for sub_id in range(1,33):
		print("processing ",sub_id)
		sub_id = "%02d" % sub_id
		sub = "s"+str(sub_id)
		file = sio.loadmat("/home/yyl/DE_CNN/DecisionTree/without_base/without_decomposed/DE_"+sub+".mat")
		X = file["data"]
		y = np.squeeze(file[arousal_or_valence+"_labels"].transpose())
		# shuffle data
		index = np.array(range(0,len(y)))
		np.random.shuffle(index)
		input_X = X[index]
		y = y[index]

		fold = 10

		acc_list = np.empty(0)
		count = 0
		
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
			train_sample = train_y.shape[0]
			test_sample = test_y.shape[0]
			# 训练模型，限制树的最大深度
			clf = DecisionTreeClassifier(max_depth=20)
			clf.fit(train_x,train_y)

			Z = clf.predict(test_x)
			accuracy = np.sum(Z==test_y)/len(Z)
			# print("fold:",curr_fold,"acc:",accuracy)
			mean_accuracy += accuracy

		# print(count,"---",key,mean_accuracy/fold*100)
		count+=1
		acc_list2 = np.append(acc_list2,mean_accuracy/fold*100)
	# print(result)
	accuracy2 = pd.DataFrame(
			{"with":acc_list1,"without":acc_list2})
	writer = pd.ExcelWriter("/home/yyl/DE_CNN/SVM/result/original_"+arousal_or_valence+".xlsx")
	accuracy2.to_excel(writer, 'result2', index=False)

	writer.save()