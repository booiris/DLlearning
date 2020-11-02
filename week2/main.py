import numpy as np
import matplotlib.pyplot as plt
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


np.random.seed(1)
X, Y = load_planar_dataset()
plt.scatter(X[0,:],X[1,:],c=Y,s=30,cmap=plt.cm.Spectral)
# plt.show()


