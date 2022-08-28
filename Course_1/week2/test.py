# 设置参数
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import psutil
# %reload_ext autoreload
# %autoreload 2

imgDir = './week2/images/resized/'
savedDataDir = './week2/SavedTraining'
saveDataFilePath = './week2/savedTrainingData128.bin'
labels_path = './week2/labels_final.csv'

loadLastTrainData = True
sliceRandomSeed = -1
trainSetPercent = 0.8
learning_rate = 0.01
num_loops = 10
num_iterations = 100

if sliceRandomSeed > 0: np.random.seed = sliceRandomSeed

import utils

labels= pd.read_csv(labels_path)
print('data set count:', labels.shape[0])

memUsage = psutil.virtual_memory()
print('memory used %.3f GB of %.3f GB | %.0f %%' % (memUsage.used / 1024 / 1024 / 1024, memUsage.total / 1024 / 1024 / 1024, memUsage.percent))
#-----------------------------------------

trainSetCount = math.floor(labels.shape[0] * trainSetPercent)
if trainSetCount < 1: trainSetCount = 1
testSetCount = labels.shape[0] - trainSetCount

print('train set count:', trainSetCount)
print('test set count:', testSetCount)

imgSize = 0

shuffledDataSet = np.array(labels)
np.random.shuffle(shuffledDataSet)
trainSetRange = shuffledDataSet[:trainSetCount, :]
testSetRange = shuffledDataSet[trainSetCount:trainSetCount + testSetCount, :]

trainSetX = []

for row in trainSetRange:
    img = plt.imread(imgDir + str(row[0]) + '.jpg')
    if imgSize == 0: imgSize = img.size
    elif imgSize != img.size:
        raise ValueError("图片尺寸不一致")
    imgArray = np.array(img)
    imgTrans = imgArray.reshape((1, img.size)).T
    trainSetX.append(imgTrans)

print('image size:', imgSize)

# 构造训练集
trainSetX = np.array(trainSetX)
trainSetX = trainSetX.squeeze().T / 2550
trainSetY = trainSetRange[:,1:].T

# 构造测试集
testSetX = []

for row in testSetRange:
    img = plt.imread(imgDir + str(row[0]) + '.jpg')
    if imgSize != img.size:
        raise ValueError("图片尺寸不一致")
    imgArray = np.array(img)
    imgTrans = imgArray.reshape((1, img.size)).T
    testSetX.append(imgTrans)

testSetX = np.array(testSetX).squeeze().T / 2550
testSetY = testSetRange[:,1:].T

memUsage = psutil.virtual_memory()
print('memory used %.3f GB of %.3f GB | %.0f %%' % (memUsage.used / 1024 / 1024 / 1024, memUsage.total / 1024 / 1024 / 1024, memUsage.percent))

if loadLastTrainData: loadedData = utils.loadFromeFile(saveDataFilePath)
else: loadedData = None
if loadedData is None:
    loadedData = {
        'w': np.zeros((trainSetX.shape[0],1), dtype="float64"),
        'b': 0.,
        'testAccuracy': 0.,
        'costs': [0.]
    }
else:
    print('last train accuracy: %.2f %%' % loadedData['trainAccuracy'])
    print('last test accuracy: %.2f %%' % loadedData['testAccuracy'])


w = loadedData['w']
b = loadedData['b']
lastCost = loadedData['costs'][-1]
lastTestAccuracy = loadedData['testAccuracy']
#-----------------------------------------
# 准备数据
# imgSize = 0


# trainSetCount = math.floor(labels.shape[0] * trainSetPercent)
# if trainSetCount < 1: trainSetCount = 1
# testSetCount = labels.shape[0] - trainSetCount

# print('train set count:', trainSetCount)
# print('test set count:', testSetCount)


# dataSet = []
# for row in labels.iterrows():
#     img = plt.imread(imgDir + str(row[1]['user_id']) + '.jpg')
#     if imgSize == 0: imgSize = img.size
#     elif imgSize != img.size:
#         raise ValueError("图片尺寸不一致")
#     imgArray = np.vstack([np.array(img, dtype="float64").reshape((1, img.size)).T / 2550, row[1]['gender']])
#     dataSet.append(imgArray)
# del labels

# dataSet = np.array(dataSet, dtype='float64').squeeze()
# print(type(dataSet))
# print(dataSet.shape)

# memUsage = psutil.virtual_memory()
# print('memory used %.3f GB of %.3f GB | %.0f %%' % (memUsage.used / 1024 / 1024 / 1024, memUsage.total / 1024 / 1024 / 1024, memUsage.percent))


# np.random.shuffle(dataSet)
# print(dataSet.shape)

# memUsage = psutil.virtual_memory()
# print('memory used %.3f GB of %.3f GB | %.0f %%' % (memUsage.used / 1024 / 1024 / 1024, memUsage.total / 1024 / 1024 / 1024, memUsage.percent))

# dataSetTrans = dataSet.T

# trainSetX = dataSetTrans[:-1, :trainSetCount]
# trainSetY = dataSetTrans[-1:, :trainSetCount]#.astype('int64')
# testSetX = dataSetTrans[:-1, trainSetCount:]
# testSetY = dataSetTrans[-1:, trainSetCount:]#.astype('int64')

# if loadLastTrainData: loadedData = utils.loadFromeFile(saveDataFilePath)
# else: loadedData = None
# if loadedData is None:
#     loadedData = {
#         'w': np.zeros((dataSet.shape[0],1), dtype="float64"),
#         'b': 0.,
#         'testAccuracy': 0.,
#         'costs': [0.]
#     }
# else:
#     print('last train accuracy: %.2f %%' % loadedData['trainAccuracy'])
#     print('last test accuracy: %.2f %%' % loadedData['testAccuracy'])


# w = loadedData['w']
# b = loadedData['b']

# memUsage = psutil.virtual_memory()
# print('memory used %.3f GB of %.3f GB | %.0f %%' % (memUsage.used / 1024 / 1024 / 1024, memUsage.total / 1024 / 1024 / 1024, memUsage.percent))
#-----------------------------------------
print('trainSetX.shape', trainSetX.shape)
print('trainSetY.shape', trainSetY.shape)
print('testSetX.shape', testSetX.shape)
print('testSetY.shape', testSetY.shape)
print('w.shape:', w.shape)
print('trainSetX data type:', trainSetX.dtype)
print('trainSetY data type:', trainSetY.dtype)
print('testSetX data type:', testSetX.dtype)
print('testSetY data type:', testSetY.dtype)
print('w data type:', w.dtype)
# print('b data type:', b.dtype)

import logicRegression
memUsage = psutil.virtual_memory()
print('memory used %.3f GB of %.3f GB | %.0f %%' % (memUsage.used / 1024 / 1024 / 1024, memUsage.total / 1024 / 1024 / 1024, memUsage.percent))
print('start train')
result = logicRegression.modelWithInitialWB(
    trainSetX, trainSetY,
    testSetX, testSetY,
    w, b,
    num_iterations, learning_rate,
    cost_record_cnt = 100, print_cost = True)


# 保存模型
saveData = {
    'w': result['w'],
    'b': result['b'],
    'costs': result['costs'],
    'trainAccuracy': result['trainAccuracy'],
    'testAccuracy': result['testAccuracy']

}
utils.save2File(saveData, saveDataFilePath)
print('save to', saveDataFilePath, 'done')









