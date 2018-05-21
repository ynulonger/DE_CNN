import numpy as np
import random
import scipy.io as sio
import matplotlib.pyplot as plt

file = sio.loadmat("../data_preprocessed_matlab/s01.mat")
# print(file.keys())
length = 1500
data = file["data"][0,:32,:length]
x = [i for i in range(0,length)]

plt.figure()
channels =16
for i in range(1,channels):
	plt.subplot(channels,1,i)
	plt.axis('off')
	lines=plt.plot(x,data[i-1,:], c=[random.random(),random.random(),random.random()])
	lines[0].set_linewidth(1.5) 
plt.show()