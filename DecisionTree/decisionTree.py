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
		dictionary = {"band1":[1],"band2":[2],"band3":[3],"band4":[4],
					  "band12":[1,2],"band13":[1,3],"band14":[1,4],"band23":[2,3],"band24":[2,4],"band34":[3,4],
					  "band123":[1,2,3],"band124":[1,2,4],"band134":[1,3,4],
					  "band1234":[1,2,3,4]}
		acc_list = np.empty(0)
		count = 0
		
		for key in sorted(dictionary.keys()):
			mean_accuracy = 0
			feature_index = get_features(dictionary[key])
			X = input_X[:,feature_index]
			# print("data shape:",X.shape)
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
			acc_list = np.append(acc_list,mean_accuracy/fold*100)
			# print(len(acc_list))
		# order: θ,α,β,γ,θ+α,θ+β,θ+γ,α+β,α+γ,β+γ,θ+α+β,θ+α+γ,α+β+γ,θ+α+β+γ
		acc_list = acc_list[[0,8,11,13,1,5,7,9,10,12,2,4,6,3]]
		result = np.vstack([result,acc_list])
		print(acc_list)
	print(result)
	accuracy = pd.DataFrame(
			{"θ":result[:,0],"α":result[:,1],"β":result[:,2],"γ":result[:,3],
			"θ+α":result[:,4],"θ+β":result[:,5],"θ+γ":result[:,6],"α+β":result[:,7],"α+γ":result[:,8],"β+γ":result[:,9],
			"θ+α+β":result[:,10],"θ+α+γ":result[:,11],"α+β+γ":result[:,12],
			"θ+α+β+γ":result[:,13]})
	writer = pd.ExcelWriter("/home/yyl/DE_CNN/DecisionTree/result/"+dir_path[2:]+arousal_or_valence+".xlsx")
	accuracy.to_excel(writer, 'result', index=False)
	writer.save()