import numpy as np

from planar_utils import sigmoid


def init_parameters(n_x, n_h, n_y):
    np.random.seed(2)

    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2
    }

    return parameters


def forward_propagation(X, parameters):
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]

    Z1 = np.dot(w1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(w2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return cache


def calculate_cost(A, Y):
    m = Y.shape[1]

    logprobs = np.multiply(np.log(A), Y) + np.multiply(np.log(1 - A), 1 - Y)
    cost = -1 / m * np.sum(logprobs)

    cost = np.squeeze(cost)

    return cost


def back_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    w2 = parameters["w2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(w2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dw1": dW1,
        "db1": db1,
        "dw2": dW2,
        "db2": db2
    }

    return grads


def update_prameters(parmeters, grads, learning_rate=1.2):
    w1 = parmeters["w1"]
    b1 = parmeters["b1"]
    w2 = parmeters["w2"]
    b2 = parmeters["b2"]
    dw1 = grads["dw1"]
    db1 = grads["db1"]
    dw2 = grads["dw2"]
    db2 = grads["db2"]

    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2

    parameters = {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2
    }

    return parameters


def train(X, Y, n_h, learning_rate, learning_iterations=10000, ):
    n_x = X.shape[0]
    n_y = Y.shape[0]

    parameters = init_parameters(n_x, n_h, n_y)

    for i in range(0, learning_iterations):

        cache = forward_propagation(X, parameters)

        cost = calculate_cost(cache["A2"], Y)

        grads = back_propagation(parameters, cache, X, Y)

        parameters = update_prameters(parameters, grads,learning_rate)

        if i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters
