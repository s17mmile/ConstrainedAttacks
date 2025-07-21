import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

from sklearn.model_selection import train_test_split

from Helpers.MNIST.Visualization import compare_MNIST784



# Specify which attack's results to use
# method = "RDSA"
method = "FGSM"
# method = "PGD"



originalDatasetPath = "Datasets/MNIST/train_data.npy"
originalLabelPath = "Datasets/MNIST/train_target.npy"

perturbedDatasetPath = "Datasets/MNIST/" + method + "_train_data.npy"
perturbedLabelPath = "Datasets/MNIST/" + method + "_train_labels.npy"

if __name__ == "__main__":
    X = np.load(originalDatasetPath, allow_pickle=True)
    Y = np.load(originalLabelPath, allow_pickle=True)

    X_attacked = np.load(perturbedDatasetPath, allow_pickle=True)
    Y_attacked = np.load(perturbedLabelPath, allow_pickle=True)

    print(X.shape)

    print("Indices: 0-" + str(Y_attacked.shape[0]-1))

    while True:
        index = input("Image index:")
        try:
            index = int(index)
        except:
            break

        compare_MNIST784(X[index], Y[index], X_attacked[index], Y_attacked[index], index)