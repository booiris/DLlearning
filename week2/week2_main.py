import matplotlib.pyplot as plt
import numpy as np

from planar_utils import plot_decision_boundary, load_planar_dataset
from week2_Predict import predict
from week2_Train import train

np.random.seed(1)
X, Y = load_planar_dataset()
# plt.scatter(X[0, :], X[1, :], c=Y, s=20, cmap=plt.cm.Spectral)
# plt.show()

parameters = train(X, Y, 4, learning_rate=1.2, learning_iterations=10000)

plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

predictions = predict(parameters, X)
print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
plt.show()
