import numpy as np
import os
import sys

sys.path.append(os.getcwd())

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

import tensorflow as tf
import keras

from Attacks.attack_dispatch import AttackDispatcher



# Rescale an arrary linearly from its original range into a given one.
def linearRescale(array, newMin, newMax):
    minimum, maximum = np.min(array), np.max(array)
    m = (newMax - newMin) / (maximum - minimum)
    b = newMin - m * minimum
    scaledArray = m * array + b
    # Remove rounding errors by clipping. The difference is tiny.
    return np.clip(scaledArray, newMin, newMax)

# Todo write constrainer that re-scales everything linearly instead of just clipping it.
def constrainer(example):
    return linearRescale(example,0,1)


# This extra specifier is necessary to use multiprocessing without getting a recursion error.
if __name__ == "__main__":

    AttackDispatcher(
        attack_type="PGD",
        datasetPath="Datasets/ImageNet/threshold_data.npy",
        targetPath="Datasets/ImageNet/threshold_target.npy",
        modelPath="Models/ImageNet/base_model.keras",
        adversaryPath="Adversaries/ImageNet/PGD_threshold_data.npy",
        newLabelPath="Adversaries/ImageNet/PGD_threshold_labels.npy",
        lossObject=keras.losses.CategoricalCrossentropy(),
        stepcount=20,
        stepsize=0.01,
        n=1024,
        workercount=8,
        chunksize=64,
        constrainer=constrainer,
        force_overwrite=False
    )
