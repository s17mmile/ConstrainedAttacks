import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

from sklearn.model_selection import train_test_split

from Helpers.MNIST.Visualization import compare_MNIST784

trainTestSplitSeed = 42

originalDatasetPath = "Datasets/MNIST/MNIST784_data.npy"
originalLabelPath = "Datasets/MNIST/MNIST784_target.npy"

# Specify which attack's results to use
# method = "RDSA"
# method = "FGSM"
method = "PGD"



perturbedDatasetPath = "Datasets/MNIST/MNIST784_adv_" + method + "_data.npy"
perturbedLabelPath = "Datasets/MNIST/MNIST784_adv_" + method + "_labels.npy"

if __name__ == "__main__":
    X = np.load(originalDatasetPath, allow_pickle=True)
    Y = np.load(originalLabelPath, allow_pickle=True)

    # Perform train-test-split to make sure we match the correct modified examples to the correct originals.
    # For this to work, the splitting seed needs to be the same as during the attack.
    # (TODO make this nicer? Maybe save train and test data separately on the device.)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=trainTestSplitSeed)

    X_test_attacked = np.load(perturbedDatasetPath, allow_pickle=True)
    Y_test_attacked = np.load(perturbedLabelPath, allow_pickle=True)

    print("Indices: 0-" + str(X_test_attacked.shape[0]-1))

    while True:
        index = input("Image index:")
        try:
            index = int(index)
        except:
            break

        compare_MNIST784(X_test[index], Y_test[index], X_test_attacked[index], Y_test_attacked[index], index)