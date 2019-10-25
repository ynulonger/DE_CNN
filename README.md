# Code for ICONIP 2018 submission
This repository contains the tensorflow implementation for our ICONIP-2018 paper: "[Continuous Convolutional Neural Network with 3D Input for EEG-Based Emotion Recognition](https://link.springer.com/chapter/10.1007/978-3-030-04239-4_39)"
## About the paper
* Title: [Continuous Convolutional Neural Network with 3D Input for EEG-Based Emotion Recognition](https://link.springer.com/chapter/10.1007/978-3-030-04239-4_39)
* Authors: [Yilong Yang](https://ynulonger.github.io/), Qingfeng Wu, YazhenFu, Xiaowei Chen
* Institution: Xiamen University
* Published in: 2018 International Conference on Neural Information Processing (ICONIP) 
## Instructions
* Before running the code, please download the DEAP dataset, unzip it and place it into the right directory. The dataset can be found [here](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html).
* Please run the get_1D_data.py to compute the **Differential Entropy** for each original .mat file. DE features of each .mat file will be stored in 1D_dataset folder.
* 1D_to_3D.py is used to transform the 1-dimentional data into 3-dimentional format, which will be used to train the proposed model.
* Using cnn.py to train and test the model (10-fold cross-validation), result of each fold will be saved in a .xls file (you can find these .xls files in ./result folder).
* count_accuracy.py is used to summarize the final accuracy of the model. The generated .xls files can be found in ./result/summary folder.
## Requirements
+ Pyhton 3
+ scipy
+ numpy
+ pandas
+ sk-learn
+ tensorflow (1.4 version)
+ import xlrd
+ import xlwt

If you have any questions, please contact yilongyang@stu.xmu.edu.cn
