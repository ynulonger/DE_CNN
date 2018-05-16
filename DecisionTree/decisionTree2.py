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
	dir_path = args[1]
	result = np.empty([0,14])
	acc_list = np.empty([0])
	for sub_id in range(1,33):
		arousal_or_valence = args[2]
		sub_id = "%02d" % sub_id
		sub = "s"+str(sub_id)
		file = sio.loadmat(dir_path+"/DE_"+sub+".mat")
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
		mean_accuracy = mean_accuracy/fold*100
		acc_list = np.append(acc_list,mean_accuracy)
		print("sub:",sub,"acc:",mean_accuracy)

	accuracy = pd.DataFrame({"original":acc_list})
	writer = pd.ExcelWriter("/home/yyl/DE_CNN/DecisionTree/result/"+dir_path[2:]+arousal_or_valence+"_original.xlsx")
	accuracy.to_excel(writer, 'result', index=False)
	writer.save()