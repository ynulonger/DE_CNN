import numpy as np
import scipy.io as sio
import pandas as pd
import sklearn.feature_selection as f_s
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

#选择K个最好的特征，返回选择特征后的数据
with_base = sio.loadmat("./with_base/DE_s01.mat")
without_base = sio.loadmat("./without_base/DE_s01.mat")

with_base_data = with_base["data"]
without_base_data = without_base["data"]
final_data = np.hstack([with_base_data,without_base_data])
print(final_data.shape)
labels = with_base["arousal_labels"].transpose()
labels = np.squeeze(labels)

def get_band_index(band):
	gamma_index = np.array(range(band,128,4))
	gamma_index = list(map(int,gamma_index))
	return gamma_index

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


with_result_1 = get_mutual_info(with_base_data[:,get_band_index(0)],labels)
without_result_1 = get_mutual_info(without_base_data[:,get_band_index(0)],labels)

with_result_2 = get_mutual_info(with_base_data[:,get_band_index(1)],labels)
without_result_2 = get_mutual_info(without_base_data[:,get_band_index(1)],labels)

with_result_3 = get_mutual_info(with_base_data[:,get_band_index(2)],labels)
without_result_3 = get_mutual_info(without_base_data[:,get_band_index(2)],labels)

with_result_4 = get_mutual_info(with_base_data[:,get_band_index(3)],labels)
without_result_4 = get_mutual_info(without_base_data[:,get_band_index(3)],labels)

data = pd.DataFrame({"with_1":with_result_1,"without_1":without_result_1,
					 "with_2":with_result_2,"without_2":without_result_2,
					 "with_3":with_result_3,"without_3":without_result_3,
					 "with_4":with_result_4,"without_4":without_result_4,
					 })
writer = pd.ExcelWriter("mutual_info.xlsx")
data.to_excel(writer,"result",index=False)
writer.save()