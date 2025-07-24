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

model = keras.models.load_model("Models/MNIST/maxpool_model.keras")

originalDatasetPath = "Datasets/MNIST/train_data.npy"
originalTargetPath = "Datasets/MNIST/train_target.npy"

perturbedDatasetPath = "Adversaries/MNIST/scaled_boxed/" + method + "_train_data.npy"
perturbedLabelPath = "Adversaries/MNIST/scaled_boxed/" + method + "_train_labels.npy"

if __name__ == "__main__":
    X = np.load(originalDatasetPath, allow_pickle=True)
    Y = np.load(originalTargetPath, allow_pickle=True)

    X_attacked = np.load(perturbedDatasetPath, allow_pickle=True)
    Y_attacked = np.load(perturbedLabelPath, allow_pickle=True)

    print()
    print("Indices: 0-" + str(Y_attacked.shape[0]-1))

    while True:
        index = input("\nImage index:")
        try:
            index = int(index)
        except:
            break

        originalLabel = model(np.array([X[index]], training = False))[0]
        compare_MNIST784(X[index], originalLabel, Y[index], X_attacked[index], Y_attacked[index], index)