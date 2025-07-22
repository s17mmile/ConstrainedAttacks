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
    example = example.numpy()[0]
    linearRescale(example,0,1)
    example = tf.convert_to_tensor(np.array([example]))
    return example


# This extra specifier is necessary to use multiprocessing without getting a recursion error.
if __name__ == "__main__":

    AttackDispatcher(
        attack_type="FGSM",
        datasetPath="Datasets/CIFAR10/train_data.npy",
        targetPath="Datasets/CIFAR10/train_target.npy",
        modelPath="Models/CIFAR10/base_model.keras",
        adversaryPath="Adversaries/CIFAR10/FGSM_train_data.npy",
        newLabelPath="Adversaries/CIFAR10/FGSM_train_labels.npy",
        lossObject=keras.losses.CategoricalCrossentropy(),
        epsilon=0.1,
        n=1024,
        workercount=8,
        chunksize=64,
        constrainer=constrainer,
        force_overwrite=True
    )
