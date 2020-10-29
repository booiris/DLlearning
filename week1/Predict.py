import numpy as np
from Train import sigmoid


def predict(X, w, b):
    w = w.reshape(X.shape[0], 1)
    Y = sigmoid(np.dot(w.T, X) + b)
    for i in range(Y.shape[1]):
        Y[0, i] = 0 if Y[0, i] <= 0.5 else 1

    return Y
