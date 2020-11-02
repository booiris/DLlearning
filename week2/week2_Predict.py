from week2_Train import forward_propagation


def predict(parameters, X):
    cache = forward_propagation(X, parameters)
    predictions = ~(cache["A2"] <= 0.5)

    return predictions
