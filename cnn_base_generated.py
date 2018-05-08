#!/usr/bin/env python
#-*- coding: utf-8 -*-
#@file: para_cnn_rnn.py
#@author: yyl
#@time: 2017/12/23 13:27
# If this runs wrong, don't ask me, I don't know why. 
# If this runs right, thank god, and I don't know why.
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

def minus(item):
    return item-1
    
# input_channel_num = 4
conv_fuse = "plus"

conv_1_shape = '4*4*1*16'
pool_1_shape = 'None'

# conv_2_shape = 'None'
conv_2_shape = '4*4*1*32'
pool_2_shape = 'None'

# conv_3_shape = 'None'
conv_3_shape = '4*4*1*64'
pool_3_shape = 'None'

conv_4_shape = '1*1*128*4'
pool_4_shape = 'None'

window_size = 1
# convolution full connected parameter
fc_size = 1024

dropout_prob = 0.5
np.random.seed(3)

calibration = 'N'
norm_type = '2D'
regularization_method = 'dropout'
enable_penalty = True

# input_file    =sys.argv[1]
# arousal_or_valence    =sys.argv[2]
# band = int(sys.argv[3])
# band_1 = int(sys.argv[4])
# band_2 = int(sys.argv[5])
# band_3 = int(sys.argv[6])

args = sys.argv[:]
input_file = args[1]
arousal_or_valence = args[2]
inputs = list(map(int,args[3:]))
bands = list(map(minus,inputs))
print(bands)
input_channel_num = len(bands)

dataset_dir = "/home/yyl/DE_CNN/DE_dataset/without_base/DE_"
###load training set

data_file = sio.loadmat(dataset_dir+input_file+".mat")
cnn_datasets = data_file["data"]
label_key = arousal_or_valence+"_labels"
labels = data_file[label_key]
labels = np.squeeze(np.transpose(labels))
print("loaded shape:",labels.shape)
lables_backup = labels
print("cnn_dataset shape before reshape:", np.shape(cnn_datasets))
cnn_datasets = cnn_datasets.transpose(0,2,3,1)

cnn_datasets = cnn_datasets[:,:,:,bands]

cnn_datasets = cnn_datasets.reshape(len(cnn_datasets), window_size, 9,9,input_channel_num)
print("cnn_dataset shape after reshape:", np.shape(cnn_datasets))
one_hot_labels = np.array(list(pd.get_dummies(labels)))
print("one_hot_labels:",one_hot_labels.shape)
labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
# shuffle data
index = np.array(range(0, len(labels)))
np.random.shuffle( index)

cnn_datasets   = cnn_datasets[index]
labels  = labels[index]


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

batch_size = 100


# kernel parameter
kernel_height_1st = 4
kernel_width_1st = 4

kernel_height_2nd = 4
kernel_width_2nd = 4

kernel_height_3rd = 4
kernel_width_3rd = 4

kernel_height_4th = 1
kernel_width_4th = 1

kernel_stride = 1
conv_channel_num = 32
# pooling parameter
pooling_height = 2
pooling_width = 2
pooling_stride = 2
# algorithn parameter
learning_rate = 1e-4


def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)

def conv2d(x, W, kernel_stride):
    # API: must strides[0]=strides[4]=1
    return tf.nn.conv2d(x, W, strides=[1, kernel_stride, kernel_stride, 1], padding='SAME')

def apply_conv1d(x, filter_width, in_channels, out_channels, kernel_stride,name):
    weight = weight_variable([filter_width, in_channels, out_channels],name)
    bias = bias_variable([out_channels],name)  # each feature map shares the same weight and bias
    return tf.nn.elu(tf.add(conv1d(x, weight, kernel_stride), bias))

def apply_conv2d(x, filter_height, filter_width, in_channels, out_channels, kernel_stride,name):
    weight = weight_variable([filter_height, filter_width, in_channels, out_channels],name)
    bias = bias_variable([out_channels],name)  # each feature map shares the same weight and bias
    print("weight shape:", np.shape(weight))
    print("x shape:", np.shape(x))
    #tf.layers.batch_normalization()
    conv = tf.add(conv2d(x, weight, kernel_stride),bias)
    return tf.nn.selu(tf.layers.batch_normalization(conv))

def apply_max_pooling(x, pooling_height, pooling_width, pooling_stride):
    # API: must ksize[0]=ksize[4]=1, strides[0]=strides[4]=1
    return tf.nn.max_pool(x, ksize=[1, pooling_height, pooling_width, 1],
                          strides=[1, pooling_stride, pooling_stride, 1], padding='SAME')

def apply_fully_connect(x, x_size, fc_size,name):
    fc_weight = weight_variable([x_size, fc_size],name)
    fc_bias = bias_variable([fc_size],name)
    return tf.nn.selu(tf.add(tf.matmul(x, fc_weight), fc_bias))

def apply_readout(x, x_size, readout_size,name):
    readout_weight = weight_variable([x_size, readout_size],name)
    readout_bias = bias_variable([readout_size],name)
    return tf.add(tf.matmul(x, readout_weight), readout_bias)

def data_1Dto2D(data, Y=9, X=9):
    data_2D = np.zeros([Y, X])
    print()
    data_2D[0] = (0,        0,          0,          data[0],    0,          data[16],   0,          0,          0       )
    data_2D[1] = (0,        0,          0,          data[1],    0,          data[17],   0,          0,          0       )
    data_2D[2] = (data[3],  0,          data[2],    0,          data[18],   0,          data[19],   0,          data[20])
    data_2D[3] = (0,        data[4],    0,          data[5],    0,          data[22],   0,          data[21],   0       )
    data_2D[4] = (data[7],  0,          data[6],    0,          data[23],   0,          data[24],   0,          data[25])
    data_2D[5] = (0,        data[8],    0,          data[9],    0,          data[27],   0,          data[26],   0       )
    data_2D[6] = (data[11], 0,          data[10],   0,          data[15],   0,          data[28],   0,          data[29])
    data_2D[7] = (0,        0,          0,          data[12],   0,          data[30],   0,          0,          0       )
    data_2D[8] = (0,        0,          0,          data[13],   data[14],   data[31],   0,          0,          0       )
    # return shape:9*9
    return data_2D

def generate_base_DE():
    DE_3D = np.empty([0,9,9])
    for i in range(0,len(inputs)):
        sampleNo = 32
        mu = 0
        sigma = 0.1
        np.random.seed(i)
        vector = np.random.normal(mu, sigma, sampleNo )
        DE_2D = data_1Dto2D(vector)
        DE_2D = DE_2D.reshape(-1,9,9)
        DE_3D = np.vstack([DE_3D,DE_2D])
    DE_3D = DE_3D.transpose(1,2,0)
    return DE_3D

print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Define parameters and functions End **********")

print("\n**********(" + time.asctime(time.localtime(time.time())) + ") Define NN structure Begin: **********")

# base_de = weight_variable([input_height,input_width,len(bands)],name="DE")
base_de = tf.constant(generate_base_DE(),dtype="float32")
base_weight = weight_variable([input_height,input_width,input_channel_num],"de_weight")
base_de = tf.multiply(base_de,base_weight)
# print(base_de)
# print("base_de:",base_de)
# input placeholder
cnn_in = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channel_num], name='cnn_in')
Y = tf.placeholder(tf.float32, shape=[None, n_labels], name='Y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
phase_train = tf.placeholder(tf.bool, name='phase_train')

cnn_In = tf.subtract(cnn_in,base_de)

###########################################################################################
# add cnn parallel to network
###########################################################################################
# first CNN layer
conv_1 = apply_conv2d(cnn_In, kernel_height_1st, kernel_width_1st, input_channel_num, conv_channel_num, kernel_stride,'conv1')
# pool_1 = apply_max_pooling(conv_1, pooling_height, pooling_width, pooling_stride)
print("\nconv_1 shape:", conv_1.shape)
# second CNN layer
conv_2 = apply_conv2d(conv_1, kernel_height_2nd, kernel_width_2nd, conv_channel_num, conv_channel_num * 2,
                      kernel_stride,'conv2')
# pool_2 = apply_max_pooling(conv_2, pooling_height, pooling_width, pooling_stride)
print("\nconv_2 shape:", conv_2.shape)
# third CNN layer
conv_3 = apply_conv2d(conv_2, kernel_height_3rd, kernel_width_3rd, conv_channel_num * 2, conv_channel_num * 4,
                      kernel_stride,'conv3')
print("\nconv_3 shape:", conv_3.shape)

# fully connected layer

shape = conv_3.get_shape().as_list()
conv_3_flat = tf.reshape(conv_3, [-1, shape[1] * shape[2] * shape[3]])
cnn_fc = apply_fully_connect(conv_3_flat, shape[1] * shape[2] * shape[3], fc_size,"fc")
# print("shape after cnn_full", np.shape(conv_3_shape))
# dropout regularizer
# Dropout (to reduce overfitting; useful when training very large neural network)
# We will turn on dropout during training & turn off during testing

cnn_fc_drop = tf.nn.dropout(cnn_fc, keep_prob)

# readout layer
y_ = apply_readout(cnn_fc_drop, fc_size, n_labels,'readout')
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

fold = 10
for curr_fold in range(fold):
    fold_size = cnn_datasets.shape[0]//fold
    indexes_list = [i for i in range(len(cnn_datasets))]
    indexes = np.array(indexes_list)
    split_list = [i for i in range(curr_fold*fold_size,(curr_fold+1)*fold_size)]
    split = np.array(split_list)
    cnn_test_x = cnn_datasets[split] 
    test_y = labels[split]

    split = np.array(list(set(indexes_list)^set(split_list)))
    cnn_train_x = cnn_datasets[split]
    train_y = labels[split]
    train_sample = train_y.shape[0]
    print("training examples:", train_sample)
    test_sample = test_y.shape[0]
    print("test examples    :",test_sample)
    # set train batch number per epoch
    batch_num_per_epoch = math.floor(cnn_train_x.shape[0]/batch_size)+ 1
    # set test batch number per epoch
    accuracy_batch_size = batch_size
    train_accuracy_batch_num = batch_num_per_epoch
    test_accuracy_batch_num = math.floor(cnn_test_x.shape[0]/batch_size)+ 1
    # print label
    one_hot_labels = np.array(list(pd.get_dummies(lables_backup)))
    print(one_hot_labels)

    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        train_accuracy_save = np.zeros(shape=[0], dtype=float)
        test_accuracy_save = np.zeros(shape=[0], dtype=float)
        test_loss_save = np.zeros(shape=[0], dtype=float)
        train_loss_save = np.zeros(shape=[0], dtype=float)
        for epoch in range(training_epochs):
            print("learning rate: ",learning_rate)
            cost_history = np.zeros(shape=[0], dtype=float)
            for b in range(batch_num_per_epoch):
                start = b* batch_size
                if (b+1)*batch_size>train_y.shape[0]:
                    offset = train_y.shape[0] % batch_size
                else:
                    offset = batch_size
                #offset = (b * batch_size) % (train_y.shape[0] - batch_size)
                #print("start->end:",start,"->",start+offset)
                cnn_batch = cnn_train_x[start:(start + offset), :, :, :, :]
                cnn_batch = cnn_batch.reshape(len(cnn_batch) * window_size, 9, 9, input_channel_num)
                # print("cnn_batch shape:",cnn_batch.shape)
                batch_y = train_y[start:(offset + start), :]
                _, c = session.run([optimizer, cost],
                    feed_dict={cnn_in: cnn_batch, Y: batch_y, keep_prob: 1 - dropout_prob,phase_train: True})
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
                    train_cnn_batch = train_cnn_batch.reshape(len(train_cnn_batch) * window_size, 9, 9, input_channel_num)
                    train_batch_y = train_y[start:(start + offset), :]

                    train_a, train_c = session.run([accuracy, cost],
                                                   feed_dict={cnn_in: train_cnn_batch,Y: train_batch_y, keep_prob: 1.0, phase_train: False})

                    train_loss = np.append(train_loss, train_c)
                    train_accuracy = np.append(train_accuracy, train_a)
                print("(" + time.asctime(time.localtime(time.time())) + ") Epoch: ", epoch + 1, " Training Cost: ",
                      np.mean(train_loss), "Training Accuracy: ", np.mean(train_accuracy))
                train_accuracy_save = np.append(train_accuracy_save, np.mean(train_accuracy))
                train_loss_save = np.append(train_loss_save, np.mean(train_loss))

                if(np.mean(train_accuracy)<0.8):
                    learning_rate=1e-4
                elif(0.8<np.mean(train_accuracy)<0.85):
                    learning_rate=5e-5
                elif(0.85<np.mean(train_accuracy)):
                    learning_rate=5e-6

                for j in range(test_accuracy_batch_num):
                    start = j * batch_size
                    if (j+1)*batch_size>test_y.shape[0]:
                        offset = test_y.shape[0] % batch_size
                    else:
                        offset = batch_size
                    #offset = (j * accuracy_batch_size) % (test_y.shape[0] - accuracy_batch_size)
                    test_cnn_batch = cnn_test_x[start:(offset + start), :, :, :, :]
                    test_cnn_batch = test_cnn_batch.reshape(len(test_cnn_batch) * window_size, 9, 9, input_channel_num)
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
            test_cnn_batch = test_cnn_batch.reshape(len(test_cnn_batch) * window_size, 9, 9, input_channel_num)
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
    #    os.system("mkdir -p ./result/cnn_rnn_parallel/tune_rnn_layer/" + output_dir)
        result = pd.DataFrame(
            {'epoch': range(1, epoch + 2), "train_accuracy": train_accuracy_save, "test_accuracy": test_accuracy_save,
             "train_loss": train_loss_save, "test_loss": test_loss_save})

        ins = pd.DataFrame({'conv_1': conv_1_shape, 'conv_2': conv_2_shape,'conv_3': conv_3_shape,
                            'cnn_fc': fc_size,'accuracy': np.mean(test_accuracy),
                            'keep_prob': 1 - dropout_prob,"epoch": epoch + 1, "norm": norm_type,
                            "learning_rate": learning_rate, "regularization": regularization_method,
                            "train_sample": train_sample, "test_sample": test_sample,"batch_size":batch_size}, index=[0])
    #    summary = pd.DataFrame({'class': one_hot_labels, 'recall': test_recall, 'precision': test_precision,
    #                            'f1_score': test_f1})  # , 'roc_auc':test_auc})
        # if band == 1:
        #     file_dir = "theta"
        # elif band ==2:
        #     file_dir = "alpha"
        # elif band ==3:
        #     file_dir = "beta"
        # else:
        #     file_dir = "gmma"
        file_dir = ""
        for i in inputs:
            file_dir = file_dir+str(i)
        # file_dir = str(band)+str(band_1)+str(band_2)+str(band_3)
        writer = pd.ExcelWriter("/home/yyl/DE_CNN/result/base_generated/"+file_dir+"/"+arousal_or_valence+"/"+input_file+"_"+str(curr_fold)+".xlsx")
        ins.to_excel(writer, 'condition', index=False)
        result.to_excel(writer, 'result', index=False)
    #    summary.to_excel(writer, 'summary', index=False)
        # fpr, tpr, auc
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        i = 0
        for key in one_hot_labels:
            fpr[key], tpr[key], _ = roc_curve(test_true[:, i], test_posi[:, i])
            roc_auc[key] = auc(fpr[key], tpr[key])
            roc = pd.DataFrame({"fpr": fpr[key], "tpr": tpr[key], "roc_auc": roc_auc[key]})
            roc.to_excel(writer, str(key), index=False)
            i += 1
        writer.save()
        '''
        with open("./result/cnn_rnn_parallel/tune_rnn_layer/"+output_dir+"/confusion_matrix.pkl", "wb") as fp:
            pickle.dump(confusion_matrix, fp)
        '''
        # save model
        for variable in tf.trainable_variables():
            print(variable.name,"->",variable.get_shape())
        '''
        saver = tf.train.Saver()
        saver.save(session,
                   "./result/cnn_rnn_parallel/tune_rnn_layer/" + output_dir + "/model_" + output_file)
        '''
        print("**********(" + time.asctime(time.localtime(time.time())) + ") Train and Test NN End **********\n")

