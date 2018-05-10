# -*- coding: utf-8 -*-
import math

# 代码功能：计算香农熵
from math import log #我们要用到对数函数，所以我们需要引入math模块中定义好的log函数（对数函数）

def calcShannonEnt(dataSet):#传入数据集
# 在这里dataSet是一个链表形式的的数据集
    countDataSet = len(dataSet) # 我们计算出这个数据集中的数据个数，在这里我们的值是5个数据集
    labelCounts={} # 构建字典，用键值对的关系我们表示出 我们数据集中的类别还有对应的关系
    for featVec in dataSet: #通过for循环，我们每次取出一个数据集，如featVec=[1,1,'yes']
        currentLabel = featVec[-1] # 取出最后一列 也就是类别的那一类，比如说‘yes’或者是‘no’
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    print(labelCounts) # 最后得到的结果是 {'yes': 2, 'no': 3}


    shannonEnt = 0.0 # 计算香农熵， 根据公式

    for key in labelCounts:
        prob = float(labelCounts[key])/countDataSet
        shannonEnt -= prob * log(prob,2)

    return shannonEnt
    
if (__name__=="__main__"):
	data = [1,2,1,3,2,4,5,6,7,8,1,2]
	print(calcShannonEnt(data))