import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import keras

from Helpers.ImageNet.Visualization import compare_ImageNet

# Specify which attack's results to use
method = input("Attack method (RDSA/FGSM/PGD): ")

model = keras.models.load_model("Models/ImageNet/base_model.keras")

originalDatasetPath = "Datasets/ImageNet/threshold_data.npy"
perturbedDatasetPath = "Adversaries/ImageNet/scaled/" + method + "_threshold_data.npy"
targetPath = "Datasets/ImageNet/threshold_target.npy"



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

        compare_ImageNet(X[index], originalLabel, X_attacked[index], perturbedLabel, Y[index], index)