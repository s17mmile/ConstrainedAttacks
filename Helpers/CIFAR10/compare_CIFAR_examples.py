import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import keras

from Helpers.CIFAR10.Visualization import compare_CIFAR10



# Specify which attack's results to use
method = input("Attack method (RDSA/FGSM/PGD): ")

model = keras.models.load_model("Models/CIFAR10/base_model.keras")

originalDatasetPath = "Datasets/CIFAR10/train_data.npy"
originalTargetPath = "Datasets/CIFAR10/train_target.npy"

perturbedDatasetPath = "Adversaries/CIFAR10/" + method + "_train_data.npy"
perturbedLabelPath = "Adversaries/CIFAR10/" + method + "_train_labels.npy"
successPath = "Adversaries/CIFAR10/" + method + "_fooling_success.npy"

if __name__ == "__main__":
    X = np.load(originalDatasetPath, allow_pickle=True)
    Y = np.load(originalTargetPath, allow_pickle=True)

    X_attacked = np.load(perturbedDatasetPath, allow_pickle=True)
    Y_attacked = np.load(perturbedLabelPath, allow_pickle=True)

    success = np.load(successPath, allow_pickle=True)

    print("Fooling Ratio: ")
    counts = np.unique(success, return_counts=True)

    print(" | " + str(counts[1][1]) + " / " + str(counts[1][0] + counts[1][1]) + " = " + str(counts[1][1] / (counts[1][0] + counts[1][1])) + " | ")

    print()
    print("Indices: 0-" + str(Y_attacked.shape[0]-1))

    while True:
        index = input("\nImage index:")
        try:
            index = int(index)
        except:
            break

        originalLabel = model(np.array([X[index]]))[0]
        print(originalLabel)
        compare_CIFAR10(X[index], originalLabel, Y[index], X_attacked[index], Y_attacked[index], index)