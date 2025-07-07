import numpy as np
import os
from sklearn.datasets import fetch_openml

MNIST784 = fetch_openml("MNIST_784", as_frame = False, parser="liac-arff")
np.save("Datasets/MNIST/MNIST784_data.npy", MNIST784.data)
np.save("Datasets/MNIST/MNIST784_target.npy", MNIST784.target)