import numpy as np


def sigmoid(z):  # 激活函数
    res = 1 / (1 + np.exp(-z))
    return res


def cal(w, b, X, Y):  # 计算参数
    m = X.shape[1]  # 输入数据的数量
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))  # 损失函数

    dw = 1/m*np.dot(X,(A-Y).T)
    db = 1/m*np.sum(A-Y)

    cost = np.squeeze(cost)

    grads = {
        "dw": dw,
        "db": db
    }

    return grads, cost


def train(X, Y, num_iterations = 1000, learning_rate = 0.001, show = False):
    dim = X.shape[0]  # 参数的维度

    costs = []
    w, b = np.zeros((dim, 1)), 0.0  # 逻辑回归参数

    for i in range(num_iterations):
        grads, cost = cal(w, b, X, Y)
        w = w - learning_rate * grads["dw"]
        b = b - learning_rate * grads["db"]

        if i % 100 == 0:
            costs.append(cost)
        if show and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {
        "w": w,
        "b": b
    }

    return params, costs

