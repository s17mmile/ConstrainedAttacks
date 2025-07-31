import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import keras

from Helpers.MNIST.Visualization import compare_MNIST784



# Specify which attack's results to use
method = input("Attack method (RDSA/FGSM/PGD): ")

model = keras.models.load_model("Models/MNIST/base_model.keras")

originalDatasetPath = "Datasets/MNIST/train_data.npy"
perturbedDatasetPath = "Adversaries/MNIST/scaled/" + method + "_train_data.npy"
targetPath = "Datasets/MNIST/train_target.npy"

if __name__ == "__main__":
    X = np.load(originalDatasetPath, allow_pickle=True, mmap_mode="r")
    X_attacked = np.load(perturbedDatasetPath, allow_pickle=True, mmap_mode="r")

    Y = np.load(targetPath, allow_pickle=True)

    print()
    print("Indices: 0-" + str(X_attacked.shape[0]-1))

    while True:
        index = input("\nImage index:")
        try:
            index = int(index)
        except:
            break

        originalLabel = model(np.array([X[index]]), training = False)[0]
        perturbedLabel = model(np.array([X_attacked[index]]), training = False)[0]

        compare_MNIST784(X[index], originalLabel, Y[index], X_attacked[index], perturbedLabel, index)