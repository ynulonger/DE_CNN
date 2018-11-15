import numpy as np
import scipy.io as sio
# import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from mpl_toolkits.mplot3d import Axes3D
import sklearn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import os
import pandas as pd
import tensorflow as tf
import time
import math

def PCA_draw(data_file):
    X = data_file["data"]
    Y = np.squeeze(data_file["arousal_labels"])

    pca = PCA(n_components=3)
    pca.fit(X)
    X_new = pca.transform(X)

    axes = plt.subplot(111,projection='3d')

    print(Y.shape)
    label_0_index = [index for index in range(len(Y)) if Y[index]==0]
    label_1_index = [index for index in range(len(Y)) if Y[index]==1]

    plt.scatter(X_new[label_0_index, 0], X_new[label_0_index, 1],X_new[label_0_index,2],marker='^')
    plt.scatter(X_new[label_1_index, 0], X_new[label_1_index, 1],X_new[label_1_index,2],marker='o')

    plt.show()



def minus(item):
    return item-1

# input_channel_num = 4
time_step = 1
window_size = 1
# convolution full connected parameter
fc_size = 1024

dropout_prob = 0.5
np.random.seed(3)

norm_type = '2D'
regularization_method = 'dropout'
enable_penalty = True


input_file = "s01"
arousal_or_valence = "arousal"

bands = [0,1,2,3]
print(bands)
input_channel_num = len(bands) * time_step

dataset_dir = "/home/yyl/DE_CNN/DE_dataset/with_base/DE_"
###load training set

data_file = sio.loadmat(dataset_dir+input_file+".mat")
cnn_datasets = data_file["data"]
label_key = arousal_or_valence+"_labels"
labels = data_file[label_key]

#2018-5-16 modified
label_index = [i for i in range(0,labels.shape[1],time_step)]

labels = labels[0,[label_index]]
labels = np.squeeze(np.transpose(labels))
# print("loaded shape:",labels.shape)
lables_backup = labels
# print("cnn_dataset shape before reshape:", np.shape(cnn_datasets))
cnn_datasets = cnn_datasets.transpose(0,2,3,1)
cnn_datasets = cnn_datasets[:,:,:,bands]
cnn_datasets = cnn_datasets.reshape(len(cnn_datasets)//time_step, window_size, 9,9,input_channel_num)

one_hot_labels = np.array(list(pd.get_dummies(labels)))
# print("one_hot_labels:",one_hot_labels.shape)
labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)

print("**********(" + time.asctime(time.localtime(time.time())) + ") Load and Split dataset End **********\n")
print("**********(" + time.asctime(time.localtime(time.time())) + ") Define parameters and functions Begin: **********\n")

# input parameter
n_input_ele = 32
n_time_step = window_size

input_height = 9
input_width = 9

n_labels = 2

# training parameter
lambda_loss_amount = 0.5
training_epochs = 80

batch_size = 256
kernel_stride = 1

# algorithn parameter
learning_rate = 1e-5

def conv2d(x, W, kernel_stride):
    return tf.nn.conv2d(x, W, strides=[1, kernel_stride, kernel_stride, 1], padding='SAME')

def apply_conv2d(x, weight, bias, kernel_stride,name):
    return tf.nn.selu(tf.add(conv2d(x, weight, kernel_stride),bias))

def apply_fully_connect(x, weight, bias,name):
    return tf.nn.selu(tf.add(tf.matmul(x, weight), bias))

def apply_readout(x, weight, bias,name):
    return tf.add(tf.matmul(x, weight), bias)

print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Define NN structure Begin: **********")

# load pre_trained model
model_file = sio.loadmat("./s01.mat")
print(model_file.keys())
conv_weight_1 = tf.Variable(model_file["conv1:0"])  #print("conv_weight_1:",conv_weight_1.shape)
conv_weight_2 = tf.Variable(model_file["conv2:0"])  #print("conv_weight_2:",conv_weight_2.shape)
conv_weight_3 = tf.Variable(model_file["conv3:0"])  #print("conv_weight_3:",conv_weight_3.shape)
conv_bias_1 = tf.Variable(np.reshape(model_file["conv1_1:0"].transpose(),-1))   #print("conv_bias_1:",conv_bias_1.shape)
conv_bias_2 = tf.Variable(np.reshape(model_file["conv2_1:0"].transpose(),-1))   # print("conv_bias_2:",conv_bias_2.shape)
conv_bias_3 = tf.Variable(np.reshape(model_file["conv3_1:0"].transpose(),-1))   # print("conv_bias_3:",conv_bias_3.shape)
fc_weight = tf.Variable(model_file["fc:0"]) # print("fc_weight:",fc_weight.shape)
fc_bias = tf.Variable(np.reshape(model_file["fc_1:0"].transpose(),-1))  # print("fc_bias:",fc_bias.shape)
readout_weight = tf.Variable(model_file["readout:0"])   # print("readout_weight::",readout_weight.shape)
readout_bias = tf.Variable(np.reshape(model_file["readout_1:0"].transpose(),-1))    # print("readout_bias:",readout_bias.shape)

# input placeholder
cnn_in = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channel_num], name='cnn_in')
Y = tf.placeholder(tf.float32, shape=[None, n_labels], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
phase_train = tf.placeholder(tf.bool, name='phase_train')

# add cnn parallel to network
###########################################################################################
# first CNN layer
conv_1 = apply_conv2d(cnn_in, conv_weight_1, conv_bias_1, kernel_stride,'conv11')
# second CNN layer
conv_2 = apply_conv2d(conv_1, conv_weight_2,conv_bias_2, kernel_stride,'conv22')
# third CNN layer
conv_3 = apply_conv2d(conv_2, conv_weight_3,conv_bias_3,kernel_stride,'conv3')

# fully connected layer
shape = conv_1.get_shape().as_list()
conv_1_flat = tf.reshape(conv_1, [-1, shape[1] * shape[2] * shape[3]])
shape = conv_2.get_shape().as_list()
conv_2_flat = tf.reshape(conv_2, [-1, shape[1] * shape[2] * shape[3]])
shape = conv_3.get_shape().as_list()
conv_3_flat = tf.reshape(conv_3, [-1, shape[1] * shape[2] * shape[3]])
cnn_fc = apply_fully_connect(conv_3_flat, fc_weight, fc_bias,"fc")

cnn_fc_drop = tf.nn.dropout(cnn_fc, keep_prob)

# readout layer
y_ = apply_readout(cnn_fc_drop, readout_weight,readout_bias,'readout')
y_pred = tf.argmax(tf.nn.softmax(y_), 1, name="y_pred")
y_posi = tf.nn.softmax(y_, name="y_posi")

# get correctly predicted object and accuracy
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

#tf.summary.scalar('accuracy',accuracy)

print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Define NN structure End **********")

print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Train and Test NN Begin: **********")
# run
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())
    test_accuracy = np.zeros(shape=[0], dtype=float)

    test_cnn_batch = cnn_datasets.reshape(len(cnn_datasets) * window_size, 9, 9, input_channel_num)
    test_batch_y = labels
    print(test_cnn_batch.shape)
    print(test_batch_y.shape)

    test_a = session.run(accuracy,
            feed_dict={cnn_in: test_cnn_batch,Y: test_batch_y,keep_prob: 1.0, phase_train: False})

    layer_1 = session.run(conv_1_flat,feed_dict={cnn_in: test_cnn_batch})
    layer_2 = session.run(conv_2_flat,feed_dict={cnn_in: test_cnn_batch})
    layer_3 = session.run(conv_3_flat,feed_dict={cnn_in: test_cnn_batch})
    conv_3_flat = session.run(conv_3_flat,feed_dict={cnn_in: test_cnn_batch})
    drop_out = session.run(cnn_fc_drop,feed_dict={cnn_in: test_cnn_batch,Y: test_batch_y,keep_prob: 1.0, phase_train: False})
    prediction = session.run(y_pred,feed_dict={cnn_in: test_cnn_batch,keep_prob: 1.0, phase_train: False})

    print("Test Accuracy: ", test_a, "\n")
    print("layer_1: ",layer_1.shape, "\n")
    print("layer_2: ",layer_2.shape, "\n")
    print("layer_3: ",layer_3.shape, "\n")
    print("drop_out: ",drop_out.shape, "\n")
    sio.savemat("save_s01.mat",{"labels":lables_backup,"layer_1":layer_1,"layer_2":layer_2,"layer_3":layer_3,"drop_out":drop_out})

