from __future__ import print_function
import tensorflow as tf
from xlrd import open_workbook
from xlutils.copy import copy
from xlwt import Workbook
import numpy as np
import sys
import os 
import pickle
import scipy.io as sio
import pandas as pd
import datetime
from sklearn import preprocessing

def minus(item):
    return item-1

def file_name(file_dir="/home/yyl/DE_CNN/MLP/result/"):
    for root, dirs, files in os.walk(file_dir):
        print(files)
    return files

# files = file_name()
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


def get_features(band_index):
	feature_index = np.empty(0)
	for i in band_index:
		band = np.array(range((i-1)*32,i*32))
		feature_index = np.append(feature_index,band)
	feature_index = list(map(int,feature_index))
	return feature_index

def onehot(labels):
	onehot_labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
	return onehot_labels

sub = int(sys.argv[1])
arousal_or_valence = sys.argv[2]
with_or_not = sys.argv[3]

inputs = list(map(int,sys.argv[4:]))
bands = list(map(minus,inputs))

print(bands)

subject = "%02d"%sub
print("subject",subject)
# 读取数据
file = sio.loadmat("/home/yyl/DE_CNN/1D_dataset/DE_s"+subject+".mat")
data = file["data"]

if with_or_not =="with_base":
	data = get_dataset_deviation(data,file["base_data"])

data = preprocessing.scale(data, axis=1, with_mean=True,with_std=True,copy=True)

feature_index = get_features(bands)
data = data[:,feature_index]
print("data shape:",data.shape)


labels = file[arousal_or_valence+"_labels"]
label  =  np.squeeze(labels)
# shuffle data
index = np.array(range(0, len(label)))
np.random.shuffle( index)
data = data[index]
label = onehot(label[index])

# print("training set:", len(y_train),"test set:",len(y_test))
# 设置模型参数
learning_rate = 0.01
training_epochs = 200
batch_size = 120
display_step = 200


n_input = data.shape[1]
n_hidden_1 = n_input//2
n_hidden_2 = n_input//4
n_hidden_3 = n_input//8
n_class = label.shape[1]

x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_class])


def multiplayer_perceptron(x, weight, bias):

	layer1 = tf.add(tf.matmul(x, weight['h1']), bias['h1'])
	layer1 = tf.nn.relu(layer1)
	layer2 = tf.add(tf.matmul(layer1, weight['h2']), bias['h2'])
	layer2 = tf.nn.relu(layer2)
	layer3 = tf.add(tf.matmul(layer2, weight['h3']), bias['h3'])
	layer3 = tf.nn.relu(layer3)
	# layer4 = tf.add(tf.matmul(layer3, weight['h4']), bias['h4'])
	# layer4 = tf.nn.relu(layer4)
	out_layer = tf.add(tf.matmul(layer3, weight['out']), bias['out'])

	return out_layer


weight = {
	'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1])),
	'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])), 
	'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])), 
	# 'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])), 
	'out': tf.Variable(tf.truncated_normal([n_hidden_3, n_class]))
}
bias = {
	'h1': tf.Variable(tf.truncated_normal([n_hidden_1])),
	'h2': tf.Variable(tf.truncated_normal([n_hidden_2])), 
	'h3': tf.Variable(tf.truncated_normal([n_hidden_3])), 
	# 'h4': tf.Variable(tf.truncated_normal([n_hidden_4])),
	'out': tf.Variable(tf.truncated_normal([n_class]))
}

# 建立模型
pred = multiplayer_perceptron(x, weight, bias)

# l2 regularization
lambda_loss_amount = 0.05
l2 = lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

# 定义损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)+l2)

# 优化
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 初始化所有变量
init = tf.initialize_all_variables()

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# 训练模型
start = datetime.datetime.now()


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

fold = 10

sum_acc = 0

for curr_fold in range(0,fold):
	# print("folder: ",curr_fold)
	fold_size = data.shape[0]//fold
	indexes_list = [i for i in range(len(data))]
	indexes = np.array(indexes_list)
	split_list = [i for i in range(curr_fold*fold_size,(curr_fold+1)*fold_size)]
	split = np.array(split_list)
	X_test = data[split] 
	y_test = label[split]

	split = np.array(list(set(indexes_list)^set(split_list)))
	X_train = data[split]
	y_train = label[split]
	train_sample = y_train.shape[0]
	test_sample = y_test.shape[0]
	print("current fold:",curr_fold,"training examples:", train_sample,"	test examples	:",test_sample)
	n_sample = X_train.shape[0]

	with tf.Session(config = config) as sess:
		sess.run(init)

		for epoch in range(training_epochs):
			avg_cost = 0
			total_batch = int(n_sample / batch_size)
			if epoch ==50:
				learning_rate = learning_rate/10
			if epoch ==100:
				learning_rate = learning_rate/10
			if epoch ==150:
				learning_rate = learning_rate/10
			
			for i in range(total_batch):
				_, c = sess.run([optimizer, cost], feed_dict={x: X_train[i*batch_size : (i+1)*batch_size, :], 
															  y: y_train[i*batch_size : (i+1)*batch_size, :]})
				avg_cost += c / total_batch
			
			if (epoch+1) % display_step == 0:
				# print('Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost))
				train_acc = accuracy.eval({x: X_train, y: y_train})
				test_acc = accuracy.eval({x: X_test, y: y_test})
				print('Epoch:', epoch,' Train Accuracy:', train_acc,'test Accuracy:', test_acc)

		# print('Opitimization Finished!')
		# Test
		acc = accuracy.eval({x: X_test, y: y_test})
		# print('Accuracy:', acc)
	sum_acc += acc
end = datetime.datetime.now()
mean_acc = sum_acc/fold*100


files = file_name()
save_file_name = arousal_or_valence+"_"+str(bands)+".xls"
print("acc:",mean_acc,"save path: /home/yyl/DE_CNN/MLP/result/",save_file_name)
if save_file_name not in files:
    book = Workbook(encoding='utf-8')
    sheet1 = book.add_sheet('result')

    sheet1.write(0,0,"sub")
    sheet1.write(0,1,"with")
    sheet1.write(0,2,"without")

    for i in range(1,33):
        sheet1.write(i,0,"s"+"%02d"%i)
    # 保存Excel book.save('path/文件名称.xls')
    book.save("/home/yyl/DE_CNN/MLP/result/"+save_file_name)

rexcel = open_workbook("/home/yyl/DE_CNN/MLP/result/"+save_file_name) # 用wlrd提供的方法读取一个excel文件
rows = rexcel.sheets()[0].nrows # 用wlrd提供的方法获得现在已有的行数
excel = copy(rexcel) # 用xlutils提供的copy方法将xlrd的对象转化为xlwt的对象
table = excel.get_sheet(0) # 用xlwt对象的方法获得要操作的sheet
row = rows
col = 1 if with_or_not=="with_base" else 2
print("location:",sub,col,mean_acc)
table.write(sub,col,mean_acc)
excel.save("/home/yyl/DE_CNN/MLP/result/"+save_file_name)
# print("running time:",end-start)
