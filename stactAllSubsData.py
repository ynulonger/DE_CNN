import numpy as np
import scipy.io as sio
dir_path = "./DE_dataset/2D/with_base/DE_"
store_dir = "../"
# all_data = np.empty([0,2400,4,9,9])
all_data = np.empty([0,2400,4,32])
all_arousal_labels = np.empty([0])
all_valence_labels = np.empty([0])

for i in range(1,33):
	sub = "s"+ "%02d" % i
	print("stacking sub ",sub, " now ...")
	file = sio.loadmat(dir_path+sub+".mat")
	sub_data = file["data"]
	print(sub_data.shape)
	sub_data = sub_data.reshape([1,2400,4,32])
	# sub_data = sub_data.reshape([1,2400,4,9,9])
	sub_valence_labels = np.squeeze(file["valence_labels"])
	sub_arousal_labels = np.squeeze(file["arousal_labels"])

	all_data = np.vstack([all_data,sub_data])
	all_valence_labels = np.append(all_valence_labels,sub_valence_labels)
	all_arousal_labels = np.append(all_arousal_labels,sub_arousal_labels)

# all_data = all_data.reshape([-1,4,9,9])
all_data = all_data.reshape([-1,1,4,32])
all_valence_labels = all_valence_labels.reshape(len(all_valence_labels),1).transpose()
all_arousal_labels = all_arousal_labels.reshape(len(all_arousal_labels),1).transpose()

print("data shape:",all_data.shape)
print("valence_labels shape:",all_valence_labels.shape)
print("arousal_labels shape:",all_arousal_labels.shape)

sio.savemat(store_dir+"CO_2D.mat",{"data":all_data,"valence_labels":all_valence_labels,"arousal_labels":all_arousal_labels})



