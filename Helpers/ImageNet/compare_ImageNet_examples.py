import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras

from Helpers.ImageNet.Visualization import compare_ImageNet

model = keras.applications.MobileNetV2(include_top=True, weights='imagenet')

# In the case of ImageNet, we're using a pretrained model, so ALL of our data is testing data.
originalDatasetPath = "Datasets/ImageNet/threshold_data.npy"
originalLabelPath = "Datasets/ImageNet/threshold_target.npy"

# Specify which attack's results to use
# method = "RDSA"
method = "FGSM"
# method = "PGD"

perturbedDatasetPath = "Datasets/ImageNet/" + method + "_threshold_data.npy"
perturbedLabelPath = "Datasets/ImageNet/" + method + "_threshold_labels.npy"

if __name__ == "__main__":
    X = np.load(originalDatasetPath, allow_pickle=True)
    Y = np.load(originalLabelPath, allow_pickle=True)

    X_attacked = np.load(perturbedDatasetPath, allow_pickle=True)
    Y_attacked = np.load(perturbedLabelPath, allow_pickle=True)

    print("Indices: 0-" + str(X_attacked.shape[0]-1))

    while True:
        index = input("Image index:")
        try:
            index = int(index)
        except:
            break

        originalLabel = model(np.array([X[index]]))[0]
        compare_ImageNet(X[index], originalLabel, X_attacked[index], Y_attacked[index], Y[index], index)