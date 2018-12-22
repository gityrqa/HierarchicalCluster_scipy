#-*-coding:utf-8-*-
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

f = np.loadtxt('data/L0.txt')
f_1 = f[10:,:-1]
print(f_1.shape)
pca = PCA() #保留所有成分
pca.fit(f)
a = pca.components_ #返回模型的各个特征向量
b = pca.explained_variance_ratio_ #返回各个成分各自的方差百分比(也称贡献率）
print(b)
pca = PCA(2)  #选取累计贡献率大于80%的主成分（3个主成分）
pca.fit(f_1)
low_d = pca.transform(f_1)   #降低维度
print(low_d)
plt.scatter(low_d[:,0],low_d[:,1])
plt.show()