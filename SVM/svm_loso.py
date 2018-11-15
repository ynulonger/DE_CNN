import sys
import numpy as np
import scipy.io as sio
from sklearn import svm
import pandas as pd
import datetime

if __name__ == '__main__':
	args = sys.argv[:]
	result = np.empty([0,15])
	acc_list = np.empty(0)
	X = np.empty([0,128])
	y = np.empty([0,2400])
	iteration = int(args[2])
	for sub_id in range(1,33):
		# print("processing ",sub_id)
		arousal_or_valence = args[1]
		sub_id = "%02d" % sub_id
		sub = "s"+str(sub_id)
		file = sio.loadmat("/home/yyl/DE_CNN/DecisionTree/with_base/DE_"+sub+".mat")
		# print(file["data"].shape)
		X = np.vstack([X,file["data"]])
		y = np.vstack([y,np.squeeze(file[arousal_or_valence+"_labels"])])
	y = np.reshape(y,[-1])
	print("X shape",X.shape)
	print("Y shape",y.shape)

	fold = 32

	count = 0

	mean_accuracy = 0
	# print("data shape:",X.shape)
	begin = datetime.datetime.now()

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
		
		clf = svm.SVC(cache_size=2000,max_iter=iteration)
		clf.fit(train_x,train_y)

		Z = clf.predict(test_x)
		accuracy = np.sum(Z==test_y)/len(Z)
		print("fold:",curr_fold,"acc:",accuracy)
		acc_list = np.append(acc_list,accuracy)

	# print(count,"---",key,mean_accuracy/fold*100)
	count+=1
	end = datetime.datetime.now()
	# print(len(acc_list))
	print(acc_list)
	print("time consuming:",end - begin)

	accuracy = pd.DataFrame(acc_list)
	accuracy.columns = [arousal_or_valence]
	writer = pd.ExcelWriter("/home/yyl/DE_CNN/SVM/result/"+"loso_"+arousal_or_valence+".xlsx")
	accuracy.to_excel(writer, 'result', index=False)
	writer.save()