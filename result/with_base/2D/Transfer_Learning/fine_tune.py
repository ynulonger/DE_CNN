#!/usr/bin/env python
#-*- coding: utf-8 -*-
#@author: yyl
import sys
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
import scipy.io as sio
import tensorflow as tf
import numpy as np
import time
import math

input_channel_num = 1
time_step = 1
window_size = 1
# convolution full connected parameter
fc_size = 1024

dropout_prob = 0.5
np.random.seed(3)

calibration = 'N'
norm_type = '2D'
regularization_method = 'dropout'
enable_penalty = True

input_file = "CO_2D"

args = sys.argv[:]
sub = args[1]
arousal_or_valence = args[2]

dataset_dir = "/home/yyl/"
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

cnn_datasets = cnn_datasets.reshape(len(cnn_datasets), window_size, 4,32,input_channel_num)
print("cnn_dataset shape after reshape:", np.shape(cnn_datasets))
one_hot_labels = np.array(list(pd.get_dummies(labels)))
print("one_hot_labels:",one_hot_labels.shape)
labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)

print("**********(" + time.asctime(time.localtime(time.time())) + ") Load and Split dataset End **********\n")
print("**********(" + time.asctime(time.localtime(time.time())) + ") Define parameters and functions Begin: **********\n")

# input parameter
n_input_ele = 32
n_time_step = window_size


input_height = 4
input_width = 32

n_labels = 2

# training parameter
lambda_loss_amount = 0.5
training_epochs = 80

batch_size = 256

# kernel parameter
kernel_height_1st = 4
kernel_width_1st = 4

kernel_height_2nd = 4
kernel_width_2nd = 4

kernel_height_3rd = 4
kernel_width_3rd = 4

kernel_stride = 1
conv_channel_num = 64

# algorithn parameter
learning_rate = 1e-4

def get_index():
    test_index = []
    for i in range(0,40):
        temp_index = [j for j in range(i*60,i*60+30)]
        test_index = np.append(test_index,temp_index)

    fine_tune_index = np.setxor1d([i for i in range(0,2400)],test_index)

    test_index = list(map(int,test_index))
    fine_tune_index = list(map(int,fine_tune_index))
    return test_index,fine_tune_index

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)

def conv2d(x, W, kernel_stride):
    return tf.nn.conv2d(x, W, strides=[1, kernel_stride, kernel_stride, 1], padding='SAME')

def apply_conv2d(x, weight, bias, kernel_stride,name):
    return tf.nn.selu(tf.add(conv2d(x, weight, kernel_stride),bias))

def apply_fully_connect(x, weight, bias,name):
    return tf.nn.selu(tf.add(tf.matmul(x, weight), bias))

def apply_readout(x, weight, bias,name):
    return tf.add(tf.matmul(x, weight), bias)

print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Begin Fine-Tune **********")

# load pre_trained model
model_file = sio.loadmat("./"+arousal_or_valence+"/models/CO_2D_"+sub+".mat")
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
print("\nconv_1 shape:", conv_1.shape)
# second CNN layer
conv_2 = apply_conv2d(conv_1, conv_weight_2,conv_bias_2, kernel_stride,'conv22')
print("\nconv_2 shape:", conv_2.shape)
# third CNN layer
conv_3 = apply_conv2d(conv_2, conv_weight_3,conv_bias_3,kernel_stride,'conv3')
print("\nconv_3 shape:", conv_3.shape)

# fully connected layer

shape = conv_3.get_shape().as_list()
conv_3_flat = tf.reshape(conv_3, [-1, shape[1] * shape[2] * shape[3]])
cnn_fc = apply_fully_connect(conv_3_flat, fc_weight, fc_bias,"fc")

cnn_fc_drop = tf.nn.dropout(cnn_fc, keep_prob)

# readout layer
y_ = apply_readout(cnn_fc_drop, readout_weight,readout_bias,'readout')
y_pred = tf.argmax(tf.nn.softmax(y_), 1, name="y_pred")
y_posi = tf.nn.softmax(y_, name="y_posi")

# l2 regularization
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
)

if enable_penalty:
    # cross entropy cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y) + l2, name='loss')
else:
    # cross entropy cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y), name='loss')

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# get correctly predicted object and accuracy
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y_), 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

#tf.summary.scalar('accuracy',accuracy)

print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Define NN structure End **********")

print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Train and Test NN Begin: **********")
# run
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

test_index,fine_tune_index = get_index()
subject_data_size = 2400
indexes_list = [i for i in range(len(cnn_datasets))]
indexes = np.array(indexes_list)
split_list = [i for i in range(int(sub)*subject_data_size,(int(sub)+1)*subject_data_size)]
split = np.array(split_list)
subject_data = cnn_datasets[split] 
cnn_test_x = subject_data[test_index]
subject_y = labels[split]
test_y = subject_y[test_index]

cnn_train_x = subject_data[fine_tune_index]
train_y = subject_y[fine_tune_index]

train_sample = train_y.shape[0]

# shuffle data
index = np.array(range(0, len(train_y)))
np.random.shuffle(index)

cnn_train_x   = cnn_train_x[index]
train_y  = train_y[index]

print("training examples:", train_sample)
test_sample = test_y.shape[0]
print("test examples    :",test_sample)
# set train batch number per epoch
batch_num_per_epoch = math.floor(cnn_train_x.shape[0]/batch_size)+ 1
# set test batch number per epoch
accuracy_batch_size = batch_size
train_accuracy_batch_num = batch_num_per_epoch
test_accuracy_batch_num = math.floor(cnn_test_x.shape[0]/batch_size)+ 1

with tf.Session(config=config) as session:
    session.run(tf.global_variables_initializer())
    train_accuracy_save = np.zeros(shape=[0], dtype=float)
    test_accuracy_save = np.zeros(shape=[0], dtype=float)
    test_loss_save = np.zeros(shape=[0], dtype=float)
    train_loss_save = np.zeros(shape=[0], dtype=float)
    for epoch in range(training_epochs):
        print("learning rate: ",learning_rate,"training set:",len(train_y))
        cost_history = np.zeros(shape=[0], dtype=float)
        for b in range(batch_num_per_epoch):
            start = b* batch_size
            if (b+1)*batch_size>train_y.shape[0]:
                offset = train_y.shape[0] % batch_size
            else:
                offset = batch_size
            cnn_batch = cnn_train_x[start:(start + offset), :, :, :, :]
            cnn_batch = cnn_batch.reshape(len(cnn_batch) * window_size,input_height,input_width, input_channel_num)
            # print("cnn_batch shape:",cnn_batch.shape)
            batch_y = train_y[start:(offset + start), :]
            _, c = session.run([optimizer, cost],
                               feed_dict={cnn_in: cnn_batch, Y: batch_y, keep_prob: 1 - dropout_prob,
                                          phase_train: True})
            cost_history = np.append(cost_history, c)
        if (epoch % 1 == 0):
            train_accuracy = np.zeros(shape=[0], dtype=float)
            test_accuracy = np.zeros(shape=[0], dtype=float)
            test_loss = np.zeros(shape=[0], dtype=float)
            train_loss = np.zeros(shape=[0], dtype=float)

            for i in range(train_accuracy_batch_num):
                start = i* batch_size
                if (i+1)*batch_size>train_y.shape[0]:
                    offset = train_y.shape[0] % batch_size
                else:
                    offset = batch_size
                #offset = (i * accuracy_batch_size) % (train_y.shape[0] - accuracy_batch_size)
                train_cnn_batch = cnn_train_x[start:(start + offset), :, :, :, :]
                train_cnn_batch = train_cnn_batch.reshape(len(train_cnn_batch) * window_size,input_height,input_width, input_channel_num)
                train_batch_y = train_y[start:(start + offset), :]

                train_a, train_c = session.run([accuracy, cost],
                                               feed_dict={cnn_in: train_cnn_batch,Y: train_batch_y, keep_prob: 1.0, phase_train: False})

                train_loss = np.append(train_loss, train_c)
                train_accuracy = np.append(train_accuracy, train_a)
            print("(" + time.asctime(time.localtime(time.time())) + ") Epoch: ", epoch + 1, " Training Cost: ",
                  np.mean(train_loss), "Training Accuracy: ", np.mean(train_accuracy))
            train_accuracy_save = np.append(train_accuracy_save, np.mean(train_accuracy))
            train_loss_save = np.append(train_loss_save, np.mean(train_loss))

            if(np.mean(train_accuracy)<0.7):
                learning_rate=1e-4
            elif(0.7<np.mean(train_accuracy)<0.85):
                learning_rate=5e-5
            elif(0.85<np.mean(train_accuracy)):
                learning_rate=1e-6

            for j in range(test_accuracy_batch_num):
                start = j * batch_size
                if (j+1)*batch_size>test_y.shape[0]:
                    offset = test_y.shape[0] % batch_size
                else:
                    offset = batch_size
                #offset = (j * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size)
                test_cnn_batch = cnn_test_x[start:(offset + start), :, :, :, :]
                test_cnn_batch = test_cnn_batch.reshape(len(test_cnn_batch) * window_size,input_height,input_width, input_channel_num)
                test_batch_y = test_y[start:(offset + start), :]

                test_a, test_c = session.run([accuracy, cost],
                                             feed_dict={cnn_in: test_cnn_batch,Y: test_batch_y,keep_prob: 1.0, phase_train: False})

                test_accuracy = np.append(test_accuracy, test_a)
                test_loss = np.append(test_loss, test_c)

            print("(" + time.asctime(time.localtime(time.time())) + ") Epoch: ", epoch + 1, " Test Cost: ",
                  np.mean(test_loss), "Test Accuracy: ", np.mean(test_accuracy), "\n")
            test_accuracy_save = np.append(test_accuracy_save, np.mean(test_accuracy))
            test_loss_save = np.append(test_loss_save, np.mean(test_loss))
        # reshuffle
        index = np.array(range(0, len(train_y)))
        np.random.shuffle(index)
        cnn_train_x=cnn_train_x[index]
        train_y=train_y[index]

    test_accuracy = np.zeros(shape=[0], dtype=float)
    test_loss = np.zeros(shape=[0], dtype=float)
    test_pred = np.zeros(shape=[0], dtype=float)
    test_true = np.zeros(shape=[0, 2], dtype=float)
    test_posi = np.zeros(shape=[0, 2], dtype=float)
    for k in range(test_accuracy_batch_num):
        start = k * batch_size
        if (k+1)*batch_size>test_y.shape[0]:
            offset = test_y.shape[0] % batch_size
        else:
            offset = batch_size
        #offset = (k * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size)
        test_cnn_batch = cnn_test_x[start:(offset + start), :, :, :, :]
        test_cnn_batch = test_cnn_batch.reshape(len(test_cnn_batch) * window_size,input_height,input_width, input_channel_num)
        test_batch_y = test_y[start:(offset + start), :]

        test_a, test_c, test_p, test_r = session.run([accuracy, cost, y_pred, y_posi],
                                                     feed_dict={cnn_in: test_cnn_batch,Y: test_batch_y, keep_prob: 1.0, phase_train: False})
        test_t = test_batch_y

        test_accuracy = np.append(test_accuracy, test_a)
        test_loss = np.append(test_loss, test_c)
        test_pred = np.append(test_pred, test_p)
        test_true = np.vstack([test_true, test_t])
        test_posi = np.vstack([test_posi, test_r])
    # test_true = tf.argmax(test_true, 1)
    test_pred_1_hot = np.asarray(pd.get_dummies(test_pred), dtype=np.int8)
    test_true_list = tf.argmax(test_true, 1).eval()

    print("(" + time.asctime(time.localtime(time.time())) + ") Final Test Cost: ", np.mean(test_loss),
          "Final Test Accuracy: ", np.mean(test_accuracy))
    # save result
    result = pd.DataFrame(
        {'epoch': range(1, epoch + 2), "train_accuracy": train_accuracy_save, "test_accuracy": test_accuracy_save,
         "train_loss": train_loss_save, "test_loss": test_loss_save})

    ins = pd.DataFrame({'accuracy': np.mean(test_accuracy),
                        'keep_prob': 1 - dropout_prob,"epoch": epoch + 1, "norm": norm_type,
                        "learning_rate": learning_rate, "regularization": regularization_method,
                        "train_sample": train_sample, "test_sample": test_sample,"batch_size":batch_size}, index=[0])

    writer = pd.ExcelWriter("./fine_tune_"+arousal_or_valence+"/"+"sub_"+str(sub)+".xlsx")
    ins.to_excel(writer, 'condition', index=False)
    result.to_excel(writer, 'result', index=False)
    writer.save()

    print("**********(" + time.asctime(time.localtime(time.time())) + ") Fine-Tune NN End **********\n")