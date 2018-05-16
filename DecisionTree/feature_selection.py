import sys
import numpy as np
import scipy.io as sio
import pandas as pd
import sklearn.feature_selection as f_s
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


#选择K个最好的特征，返回选择特征后的数据


def get_band_index(start,stop,step):
	index = np.array(range(start,stop,step))
	index = list(map(int,index))
	return index

def get_mutual_info(X,y):
	return f_s.mutual_info_regression(X,y)

def get_RFE():
	estimator = SVR(kernel="linear")
	selector = RFE(estimator, 32, step=1)
	selector = selector.fit(final_data, labels)
	print(selector.support_)
	return selector.ranking_

def anova(X,y):
	F_value,P_value = f_s.f_classif(X,y)
	with_F,with_P = anova(with_base_data,labels)
	without_F,without_P = anova(without_base_data,labels)

	data = pd.DataFrame({"with_F":with_F,"without_F":without_F,"with_P":with_P,"without_P":without_P})
	writer = pd.ExcelWriter("anova.xlsx")
	data.to_excel(writer,"result",index=False)
	writer.save()

def tree_based(X,y):
	print(X.shape)
	clf = ExtraTreesClassifier()
	clf = clf.fit(X, y)
	
	# print(clf.feature_importances_)
	model = SelectFromModel(clf, prefit=True)
	X_new = model.transform(X)

	weight = clf.feature_importances_
	
	return weight

result = np.empty([0,10])
for sub_id in range(1,33):
	print("processing ",sub_id)
	sub_id = "%02d" % sub_id
	with_base = sio.loadmat("./with_base/DE_s"+str(sub_id)+".mat")
	without_base = sio.loadmat("./without_base/DE_s"+str(sub_id)+".mat")
	with_original = sio.loadmat("./with_base/without_decomposed/DE_s"+str(sub_id)+".mat")
	without_original = sio.loadmat("./without_base/without_decomposed/DE_s"+str(sub_id)+".mat")

	with_base_data = with_base["data"]
	without_base_data = without_base["data"]
	with_original_data = with_original["data"]
	without_original_data = without_original["data"]
	final_data = np.hstack([with_base_data,without_base_data,with_original_data,without_original_data])

	labels = with_base["valence_labels"].transpose()
	labels = np.squeeze(labels)
	row = tree_based(final_data,labels)


	with_weight = row[:128]
	without_weight = row[128:256]
	with_original = row[256:288]
	without_original = row[288:]

	print("shape:",with_weight.shape)
	print("shape:",without_weight.shape)
	print("shape:",with_original.shape)
	print("shape:",without_original.shape)

	theta_band = get_band_index(0,128,4)
	alpha_band = get_band_index(1,128,4)
	beta_band = get_band_index(2,128,4)
	gamma_band = get_band_index(3,128,4)

	row = [np.mean(with_weight[theta_band]),np.mean(with_weight[alpha_band]),
		   np.mean(with_weight[beta_band]),np.mean(with_weight[gamma_band]),
		   np.mean(without_weight[theta_band]),np.mean(without_weight[alpha_band]),
		   np.mean(without_weight[beta_band]),np.mean(without_weight[gamma_band]),
		   np.mean(without_original),np.mean(without_original)]

	result = np.vstack([result,row])

data = pd.DataFrame({"with_theta":result[:,0],
					 "with_alpha":result[:,1],
					 "with_beta":result[:,2],
					 "with_gamma":result[:,3],
					 "without_theta":result[:,4],
					 "without_alpha":result[:,5],
					 "without_beta":result[:,6],
					 "without_gamma":result[:,7],
					 "with_original":result[:,8],
					 "without_original":result[:,9]})

print("fffffff",result.shape)

writer = pd.ExcelWriter("tree_based_total_1"+".xlsx")
data.to_excel(writer,"result",index=False)
writer.save()




