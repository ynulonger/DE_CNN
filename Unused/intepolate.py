# -*- coding: utf-8 -*-  
""" 
演示二维插值。 
"""  
import numpy as np  
from scipy import interpolate  
import scipy.io as sio
import matplotlib as mpl  

file = sio.loadmat("DE_dataset/with_base/DE_s01.mat")
segments = file["data"]
def inter(image,size):
	y,x= np.mgrid[-1:1:9j, -1:1:9j]  
	newfunc = interpolate.interp2d(x, y, image, kind='cubic')  
	xnew = np.linspace(-1,1,size)#x  
	ynew = np.linspace(-1,1,size)#y  
	fnew = newfunc(xnew, ynew) 
	return fnew

#samples
# 2400*4*9*9
size = 32
new_segments = np.empty([0,size,size,4])
for segment in segments:
	# 4*9*9
	temp_segment = np.empty([0,size,size])
	for band in segment:
		new = inter(band,size)
		new = new.reshape(-1,size,size)
		temp_segment = np.vstack([temp_segment,new])
	temp_segment = temp_segment.transpose(1,2,0)
	temp_segment = temp_segment.reshape(-1,size,size,4)
	# print(temp_segment.shape)
	new_segments = np.vstack([new_segments,temp_segment])

print(new_segments.shape,"end...")

sio.savemat("/home/yyl/DE_CNN/32*32.mat",{"data":new_segments,"valence_labels":file["valence_labels"],"arousal_labels":file["arousal_labels"]})



