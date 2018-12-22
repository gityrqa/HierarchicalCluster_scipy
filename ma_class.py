import numpy as np
import math
import pandas as pd
#import maClass
import matplotlib.pyplot as plt
np.set_printoptions(threshold= 1000000000000)

#SMA  计算简单移动平均
def sma(X,ma_list):
    b = np.zeros(len(X))
    b[0] = X[0]
    for i in range(1,len(X)):
        if i < ma_list - 1:
            b[i] = sum(X[0:i])/i
        else:
            b[i] = sum(X[i - ma_list + 1:i + 1]) / ma_list
    #print(a, b)
    return b
#对data 按照二阶斜率分类
def maSlope(X, maS= 50):
    smaS = sma(X, maS-10)
    smaS = sma(smaS, 10)
    s = X.shape[0]
    Y = np.zeros([s, 2])
    Y[1:, 0] = np.diff(smaS, 1)* 1000
    Y[2:, 1] = np.diff(smaS, 2)* 1000
    #print(Y.shape)
    Ylabel = np.zeros([s, 2])
    for i in range(1, s):
        if Y[i-1, 0] >= 0:
            Ylabel[i, 0] = 0
        else:
            Ylabel[i, 0] = 1
        if Y[i-1, 1] >= 0:
            Ylabel[i, 1] = 0
        else:
            Ylabel[i, 1] = 1
    print(Ylabel.shape)
    return Ylabel



#data 转换
def dataReshape(X, dtw= 10):
    dtwL = np.ones([len(X), dtw+1])
    for i in range(dtw+1):
        dtwL[dtw-i:, i] = X[:len(X)-(dtw-i)]
    print('dtwL.shape',dtwL.shape)
    dtwN = normalization(dtwL)
    return dtwN
# 归一化
def normalization(X):
    X_1 = X[:,0:-1]
    X_mean = np.mean(X_1, axis=1)
    X_normal = X/X_mean.reshape((len(X),1))
    return X_normal
# 2进制标签转为10进制
def translation_2to10(label_2):
    label_10 = label_2[:,0]*8+label_2[:,1]*4+label_2[:,2]*2+label_2[:,3]
    return label_10
# data 移位
def shift(label_10):
    label_10_1 = np.zeros(label_10.shape)
    label_10_1[1:]=label_10[:-1]
    return label_10_1

def main(filePath = 'GOLD.txt', reshape = 20, ma_1 = 50, ma_2 = 100):
    a = np.loadtxt(filePath, usecols=4, delimiter=',')
    a_reshape_normalization = dataReshape(a, reshape)[100:] # 数据整理 归一化
    label_2 = np.hstack((maSlope(a, ma_1), maSlope(a, ma_2)))[100:] # 给数据打标签
    label_10 = translation_2to10(label_2) # 2维数据转为10维
    label_10_1 = shift(label_10) # 标签向前移动一位
    print('label.shape:', label_10.shape) 
    #print(label_10)
    for i in range(16):
        li = np.where(label_10_1 == i)[0]
        np.savetxt("data/L%s.txt" % i, a_reshape_normalization[li])
        np.savetxt("index/L%s.txt" % i, li)


if __name__=="__main__":
    main(filePath = 'GOLD.txt', reshape = 10, ma_1 = 30, ma_2 = 90)
    total = 0
    for i in range(16):
        a = np.loadtxt('data/L%s.txt' % i)
        print('L%i' % i, a.shape)
        total = total + a.shape[0]
    print(total)
