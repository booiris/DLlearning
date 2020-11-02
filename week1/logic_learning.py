import numpy as np
from DataSet import dataset
import matplotlib.pyplot as plt
from week1_Train import train
from week1_Predict import predict

train_x, train_y, test_x, test_y, classes = dataset()  # 导入数据

# index = 10
# plt.imshow(train_x[index]) # 显示图片

# 图片压缩为向量
train_x = train_x.reshape(train_x.shape[0], -1).T
test_x = test_x.reshape(test_x.shape[0], -1).T

# 向量归一化
train_x = train_x / 255
test_x = test_x / 255

params, c = train(train_x, train_y, 2000, 0.005)
test_predict_y = predict(test_x, params["w"], params["b"])
train_predict_y = predict(train_x, params["w"], params["b"])

print("train accuracy: {} %".format(100 - np.mean(np.abs(train_predict_y - train_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(test_predict_y - test_y)) * 100))

