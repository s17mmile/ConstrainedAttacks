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
originalDatasetPath = "Datasets/ImageNet/ImageNetv2_data.npy"
originalLabelPath = "Datasets/ImageNet/ImageNetv2_target.npy"

# Specify which attack's results to use
# method = "RDSA"
method = "FGSM"
# method = "PGD"

perturbedDatasetPath = "Datasets/ImageNet/ImageNetv2_adv_" + method + "_data.npy"
perturbedLabelPath = "Datasets/ImageNet/ImageNetv2_adv_" + method + "_labels.npy"

if __name__ == "__main__":
    X = np.load(originalDatasetPath, allow_pickle=True)
    Target = np.load(originalLabelPath, allow_pickle=True)

    X_attacked = np.load(perturbedDatasetPath, allow_pickle=True)
    Y_attacked = np.load(perturbedLabelPath, allow_pickle=True)

    print("Indices: 0-" + str(X_attacked.shape[0]-1))

    while True:
        index = input("Image index:")
        try:
            index = int(index)
        except:
            break

        # Since I used one image from each category for the 1000 images test, I adjust the indices for the original data by a factor of 10 (number of pictures in each category).
        # Ideally, they're a 1:1 match, but I figured generating the full 10k images is overkill for now.
        originalLabel = model(np.array([X[10*index]]))[0]
        compare_ImageNet(X[10*index], originalLabel, X_attacked[index], Y_attacked[index], Target[10*index], index)