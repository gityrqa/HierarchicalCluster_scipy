#-*-coding:utf-8-*-
import sys
import numpy as np
def distance(x, y):
    return abs(x - y)
def dtw(XX, YY):
    X = [0 for i in range(len(XX))]
    Y = [0 for i in range(len(YY))]
    for i in range(len(XX)):
        X[i] = XX[-i-1]
    for i in range(len(YY)):
        Y[i] = YY[-i-1]
    print('X:',X)
    print('Y:',Y)
    print("Computing dtw... ...")

    l1 = len(X)
    l2 = len(Y)
    D = [[0 for i in range(l1+1)] for j in range(l2+1)]
    print(D)
    for i in range(1, l1+1):
        D[0][i] = sys.maxsize
    for j in range(1, l2+1):
        D[j][0] = sys.maxsize
    for j in range(1, l2 + 1):
        for i in range(1, l1 + 1):
            D[j][i] = distance(X[i-1], Y[j-1]) + min(D[j-1][i], D[j][i-1], D[j-1][i-1])


    T = np.array(D)
    print(T)
    T1 = np.delete(T, 0, axis=0)
    T2 = np.delete(T1, 0, axis=1)
    print(T2)
    #print(T.shape)
    dtw = min(np.min(T[:, T.shape[1]-1]), np.min(T[T.shape[0]-1, :]))
    return (dtw)



if __name__ == '__main__':
    X = [1, 2,3, 4, 5]
    Y = [1, 2,4,3,5]
    print(dtw(X, Y))
