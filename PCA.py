
'''
算法来自于《机器学习实战》 第13章 利用PCA来简化数据
'''
from numpy import*

# 数据加载函数
# 使用了两个list comprehension来构建矩阵
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

'''
pca() 函数有两个参数：第一个参数是用于进行PCA操作的数据集，第二个参数 topNfeat则是一个可选参数，即应用的N个特征。
如果不指定 topNfeat 的值，那么函数就会返回前9 999 999个特征，或者原始数据中全部的特征。
首先计算并减去原始数据集的平均值 。然后，计算协方差矩阵及其特征值，接着利用argsort() 函数对特征值进行从小到大的排序。
根据特征值排序结果的逆序就可以得到topNfeat 个最大的特征向量 。
这些特征向量将构成后面对数据进行转换的矩阵，该矩阵则利用N个特征将原始数据转换到新空间中 。
最后，原始数据被重构后返回用于调试，同时降维之后的数据集也被返回了。
'''
def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals # remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)            # sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  # cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       # reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects  # transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat
