#-*-coding:utf-8-*-
# 导入相应的包
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
import dtw  # EURUSD 实测中dtw表现 差于 euclidean

# 距离函数 用于替换 metric='euclidean'，格式 metric=distance
def distance(a, b):
    dist = np.sum((a-b)**2)
    return dist
# 累计求和函数
def cumsum(profit):
    profit_add = np.zeros(shape=(len(profit,)))
    for i in range(len(profit)):
        profit_add[i] = np.sum(profit[:i+1])
    return profit_add
# 层次聚类流程，使用了scipy.cluster.hierarchy包
def sch_distPdist_linkage_fcluster(points, t, method='complete', metric='euclidean'):
    # 1. 层次聚类
    # 生成点与点之间的距离矩阵,这里用的欧氏距离:
    disMat = sch.distance.pdist(points, metric)
    #print(disMat)
    # 进行层次聚类:
    Z = sch.linkage(disMat, method, metric)
    # # 将层级聚类结果以树状图表示出来并保存为plot_dendrogram.png
    # P = sch.dendrogram(Z)
    # plt.savefig('plot_dendrogram.png')
    # # 根据linkage matrix Z得到聚类结果:
    cluster = sch.fcluster(Z, t, criterion='maxclust')
    #print("Original cluster by hierarchy clustering:\n", cluster)

    # PCA 主成份分析 用于结果的散点图显示
    pca = PCA(2)  # 选取2个主成份
    pca.fit(points)
    low_d = pca.transform(points)  # 降低维度

    plt.figure()
    mark = ['pb', 'or', 'ob', 'og', 'ok', 'oy', 'om', 'oc',
                  'sr', 'sb', 'sg', 'sk', 'sy', 'sm', 'sc',
                  'pr', 'pb', 'pg', 'pk', 'py', 'pm', 'pc',
                  'Dr', 'Db', 'Dg', 'Dk', 'Dy', 'Dm', 'Dc']
    for i in range(len(points)):
        markIndex = int(cluster[i])%28+1  # 为样本指定颜色
        plt.plot(low_d[i, 0], low_d[i, 1], mark[markIndex], markersize=6)
        #plt.text(points[i, 0], points[i, 1], i)
    plt.grid()
    plt.show()
    plt.pause(3)
    return cluster

if __name__=='__main__':
    sum_all = 0 # 总盈利
    total = 0 # sum（所有类的长度）= z
    z = len(np.loadtxt('GOLD.txt', usecols=4, delimiter=','))-100 # z= 所有样本的总长度
    print(z)
    profit = np.zeros((z,)) # 所有样本单次交易获利情况
    for i in range(16):
        points = np.loadtxt('data/L%s.txt'%i) # 载入分类点集
        index_L = np.loadtxt('index/L%s.txt'%i) # 载入分类集中每个个体在总样本中的位置信息
        profit_i = np.zeros(shape=(len(points,))) # 分类集中单次交易获利情况
        total = total +len(points)
        print('%s points.shape:' % i, points.shape)
        t = points.shape[0]//50    # 对分类集再分类，t表示再分类的数目，保证每类平均50个点以上
        print('%s points.cluster:' % i, t)
        plt.ion()
        cluster = sch_distPdist_linkage_fcluster(points[:,:-1], t, method='complete', metric='euclidean')
        for j in range(1,t+1):
            index = np.where(cluster == j)[0] #再分类的第j类中的样本点在分类集i中的位置
            sum_j = np.sum(points[index, -1]-points[index, -2])
            sum_all = sum_all + np.abs(sum_j)
            if sum_j >=0: # 分类集i 的但次交易获利情况
                profit_i[index] = points[index, -1]-points[index, -2]
            else:
                profit_i[index] = -(points[index, -1] - points[index, -2])
        profit[index_L.astype('int64')] = profit_i # 所有样本单次交易获利情况
    mean_total = sum_all/total  # 总样本平均单次获利
    print('mean_total:',mean_total)
    #print(np.mean(profit))
    profit_add = cumsum(profit)  #总样本累积获利
    plt.figure()
    plt.plot(profit_add,'r')
    plt.grid()
    plt.show()

    plt.pause(10)
    plt.close()

# method是指计算类间距离的方法,比较常用的有3种:
# (1)single:最近邻,把类与类间距离最近的作为类间距
# (2)complete:最远邻,把类与类间距离最远的作为类间距
# (3)average:平均距离,类与类间所有pairs距离的平均



# 自定义 distance函数 需要注意以下几个方面：
# 函数传入两个参数：比如，自定义函数为：def selfDisFuc(a,b):
# 传入参数类型是<class 'numpy.ndarray'>
# 传入参数的维度必须一样
# 返回值必须是一个代表距离的数


# metric的取值如下：
#
#  braycurtis
#  canberra
#  chebyshev：切比雪夫距离
#  cityblock
#  correlation：相关系数
#  cosine：余弦夹角
#  dice
#  euclidean：欧式距离
#  hamming：汉明距离
#  jaccard：杰卡德相似系数
#  kulsinski
#  mahalanobis：马氏距离
#  matching
#  minkowski：闵可夫斯基距离
#  rogerstanimoto
#  russellrao
#  seuclidean：标准化欧式距离
#  sokalmichener
#  sokalsneath
#  sqeuclidean
#  wminkowski
#  yule



# '.'       point marker
# ','       pixel marker
# 'o'       circle marker
# 'v'       triangle_down marker
# '^'       triangle_up marker
# '<'       triangle_left marker
# '>'       triangle_right marker
# '1'       tri_down marker
# '2'       tri_up marker
# '3'       tri_left marker
# '4'       tri_right marker
# 's'       square marker
# 'p'       pentagon marker
# '*'       star marker
# 'h'       hexagon1 marker
# 'H'       hexagon2 marker
# '+'       plus marker
# 'x'       x marker
# 'D'       diamond marker
# 'd'       thin_diamond marker
# '|'       vline marker
# '_'       hline marker

# cnames = {
# 'aliceblue':            '#F0F8FF',
# 'antiquewhite':         '#FAEBD7',
# 'aqua':                 '#00FFFF',
# 'aquamarine':           '#7FFFD4',
# 'azure':                '#F0FFFF',
# 'beige':                '#F5F5DC',
# 'bisque':               '#FFE4C4',
# 'black':                '#000000',
# 'blanchedalmond':       '#FFEBCD',
# 'blue':                 '#0000FF',
# 'blueviolet':           '#8A2BE2',
# 'brown':                '#A52A2A',
# 'burlywood':            '#DEB887',
# 'cadetblue':            '#5F9EA0',
# 'chartreuse':           '#7FFF00',
# 'chocolate':            '#D2691E',
# 'coral':                '#FF7F50',
# 'cornflowerblue':       '#6495ED',
# 'cornsilk':             '#FFF8DC',
# 'crimson':              '#DC143C',
# 'cyan':                 '#00FFFF',
# 'darkblue':             '#00008B',
# 'darkcyan':             '#008B8B',
# 'darkgoldenrod':        '#B8860B',
# 'darkgray':             '#A9A9A9',
# 'darkgreen':            '#006400',
# 'darkkhaki':            '#BDB76B',
# 'darkmagenta':          '#8B008B',
# 'darkolivegreen':       '#556B2F',
# 'darkorange':           '#FF8C00',
# 'darkorchid':           '#9932CC',
# 'darkred':              '#8B0000',
# 'darksalmon':           '#E9967A',
# 'darkseagreen':         '#8FBC8F',
# 'darkslateblue':        '#483D8B',
# 'darkslategray':        '#2F4F4F',
# 'darkturquoise':        '#00CED1',
# 'darkviolet':           '#9400D3',
# 'deeppink':             '#FF1493',
# 'deepskyblue':          '#00BFFF',
# 'dimgray':              '#696969',
# 'dodgerblue':           '#1E90FF',
# 'firebrick':            '#B22222',
# 'floralwhite':          '#FFFAF0',
# 'forestgreen':          '#228B22',
# 'fuchsia':              '#FF00FF',
# 'gainsboro':            '#DCDCDC',
# 'ghostwhite':           '#F8F8FF',
# 'gold':                 '#FFD700',
# 'goldenrod':            '#DAA520',
# 'gray':                 '#808080',
# 'green':                '#008000',
# 'greenyellow':          '#ADFF2F',
# 'honeydew':             '#F0FFF0',
# 'hotpink':              '#FF69B4',
# 'indianred':            '#CD5C5C',
# 'indigo':               '#4B0082',
# 'ivory':                '#FFFFF0',
# 'khaki':                '#F0E68C',
# 'lavender':             '#E6E6FA',
# 'lavenderblush':        '#FFF0F5',
# 'lawngreen':            '#7CFC00',
# 'lemonchiffon':         '#FFFACD',
# 'lightblue':            '#ADD8E6',
# 'lightcoral':           '#F08080',
# 'lightcyan':            '#E0FFFF',
# 'lightgoldenrodyellow': '#FAFAD2',
# 'lightgreen':           '#90EE90',
# 'lightgray':            '#D3D3D3',
# 'lightpink':            '#FFB6C1',
# 'lightsalmon':          '#FFA07A',
# 'lightseagreen':        '#20B2AA',
# 'lightskyblue':         '#87CEFA',
# 'lightslategray':       '#778899',
# 'lightsteelblue':       '#B0C4DE',
# 'lightyellow':          '#FFFFE0',
# 'lime':                 '#00FF00',
# 'limegreen':            '#32CD32',
# 'linen':                '#FAF0E6',
# 'magenta':              '#FF00FF',
# 'maroon':               '#800000',
# 'mediumaquamarine':     '#66CDAA',
# 'mediumblue':           '#0000CD',
# 'mediumorchid':         '#BA55D3',
# 'mediumpurple':         '#9370DB',
# 'mediumseagreen':       '#3CB371',
# 'mediumslateblue':      '#7B68EE',
# 'mediumspringgreen':    '#00FA9A',
# 'mediumturquoise':      '#48D1CC',
# 'mediumvioletred':      '#C71585',
# 'midnightblue':         '#191970',
# 'mintcream':            '#F5FFFA',
# 'mistyrose':            '#FFE4E1',
# 'moccasin':             '#FFE4B5',
# 'navajowhite':          '#FFDEAD',
# 'navy':                 '#000080',
# 'oldlace':              '#FDF5E6',
# 'olive':                '#808000',
# 'olivedrab':            '#6B8E23',
# 'orange':               '#FFA500',
# 'orangered':            '#FF4500',
# 'orchid':               '#DA70D6',
# 'palegoldenrod':        '#EEE8AA',
# 'palegreen':            '#98FB98',
# 'paleturquoise':        '#AFEEEE',
# 'palevioletred':        '#DB7093',
# 'papayawhip':           '#FFEFD5',
# 'peachpuff':            '#FFDAB9',
# 'peru':                 '#CD853F',
# 'pink':                 '#FFC0CB',
# 'plum':                 '#DDA0DD',
# 'powderblue':           '#B0E0E6',
# 'purple':               '#800080',
# 'red':                  '#FF0000',
# 'rosybrown':            '#BC8F8F',
# 'royalblue':            '#4169E1',
# 'saddlebrown':          '#8B4513',
# 'salmon':               '#FA8072',
# 'sandybrown':           '#FAA460',
# 'seagreen':             '#2E8B57',
# 'seashell':             '#FFF5EE',
# 'sienna':               '#A0522D',
# 'silver':               '#C0C0C0',
# 'skyblue':              '#87CEEB',
# 'slateblue':            '#6A5ACD',
# 'slategray':            '#708090',
# 'snow':                 '#FFFAFA',
# 'springgreen':          '#00FF7F',
# 'steelblue':            '#4682B4',
# 'tan':                  '#D2B48C',
# 'teal':                 '#008080',
# 'thistle':              '#D8BFD8',
# 'tomato':               '#FF6347',
# 'turquoise':            '#40E0D0',
# 'violet':               '#EE82EE',
# 'wheat':                '#F5DEB3',
# 'white':                '#FFFFFF',
# 'whitesmoke':           '#F5F5F5',
# 'yellow':               '#FFFF00',
# 'yellowgreen':          '#9ACD32'}
