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

def stepsize(step):
    return 0.1*(1/np.sqrt(2))**step



# This extra specifier is necessary to use multiprocessing without getting a recursion error.
if __name__ == "__main__":

    print("\n\n\nCIFAR FGSM\n")
    AttackDispatcher(
        attack_type="FGSM",
        datasetPath="Datasets/CIFAR10/train_data.npy",
        targetPath="Datasets/CIFAR10/train_target.npy",
        modelPath="Models/CIFAR10/base_model.keras",
        adversaryPath="Adversaries/CIFAR10/FGSM_train_data.npy",
        newLabelPath="Adversaries/CIFAR10/FGSM_train_labels.npy",
        lossObject=keras.losses.CategoricalCrossentropy(),
        epsilon=0.1,
        # # n=1024,
        workercount=8,
        chunksize=64,
        constrainer=constrainer,
        force_overwrite=True
    )

    print("\n\n\nCIFAR PGD\n")
    AttackDispatcher(
        attack_type="PGD",
        datasetPath="Datasets/CIFAR10/train_data.npy",
        targetPath="Datasets/CIFAR10/train_target.npy",
        modelPath="Models/CIFAR10/base_model.keras",
        adversaryPath="Adversaries/CIFAR10/PGD_train_data.npy",
        newLabelPath="Adversaries/CIFAR10/PGD_train_labels.npy",
        lossObject=keras.losses.CategoricalCrossentropy(),
        stepcount=20,
        stepsize=stepsize,
        # # n=1024,
        workercount=8,
        chunksize=64,
        constrainer=constrainer,
        force_overwrite=True
    )

    print("\n\n\nCIFAR RDSA\n")
    AttackDispatcher(
        attack_type="RDSA",
        datasetPath="Datasets/CIFAR10/train_data.npy",
        targetPath="Datasets/CIFAR10/train_target.npy",
        modelPath="Models/CIFAR10/base_model.keras",
        adversaryPath="Adversaries/CIFAR10/RDSA_train_data.npy",
        newLabelPath="Adversaries/CIFAR10/RDSA_train_labels.npy",
        attempts=25,
        categoricalFeatureMaximum=100,
        binCount=100,
        perturbedFeatureCount=300,
        # # n=1024,
        workercount=8,
        chunksize=64,
        constrainer=constrainer,
        force_overwrite=True
    )



    print("\n\n\nImageNet FGSM\n")
    AttackDispatcher(
        attack_type="FGSM",
        datasetPath="Datasets/ImageNet/threshold_data.npy",
        targetPath="Datasets/ImageNet/threshold_target.npy",
        modelPath="Models/ImageNet/base_model.keras",
        adversaryPath="Adversaries/ImageNet/FGSM_threshold_data.npy",
        newLabelPath="Adversaries/ImageNet/FGSM_threshold_labels.npy",
        lossObject=keras.losses.CategoricalCrossentropy(),
        epsilon=0.1,
        # n=1024,
        workercount=8,
        chunksize=64,
        constrainer=constrainer,
        force_overwrite=True
    )

    print("\n\n\nImageNet PGD\n")
    AttackDispatcher(
        attack_type="PGD",
        datasetPath="Datasets/ImageNet/threshold_data.npy",
        targetPath="Datasets/ImageNet/threshold_target.npy",
        modelPath="Models/ImageNet/base_model.keras",
        adversaryPath="Adversaries/ImageNet/PGD_threshold_data.npy",
        newLabelPath="Adversaries/ImageNet/PGD_threshold_labels.npy",
        lossObject=keras.losses.CategoricalCrossentropy(),
        stepcount=20,
        stepsize=stepsize,
        # n=1024,
        workercount=8,
        chunksize=64,
        constrainer=constrainer,
        force_overwrite=True
    )

    print("\n\n\nImageNet RDSA\n")
    AttackDispatcher(
        attack_type="RDSA",
        datasetPath="Datasets/ImageNet/threshold_data.npy",
        targetPath="Datasets/ImageNet/threshold_target.npy",
        modelPath="Models/ImageNet/base_model.keras",
        adversaryPath="Adversaries/ImageNet/RDSA_threshold_data.npy",
        newLabelPath="Adversaries/ImageNet/RDSA_threshold_labels.npy",
        attempts=25,
        categoricalFeatureMaximum=150,
        binCount=100,
        perturbedFeatureCount=2000,
        # n=1024,
        workercount=8,
        chunksize=64,
        constrainer=constrainer,
        force_overwrite=True
    )



    print("\n\n\nMNIST FGSM\n")
    AttackDispatcher(
        attack_type="FGSM",
        datasetPath="Datasets/MNIST/train_data.npy",
        targetPath="Datasets/MNIST/train_target.npy",
        modelPath="Models/MNIST/maxpool_model.keras",
        adversaryPath="Adversaries/MNIST/FGSM_train_data.npy",
        newLabelPath="Adversaries/MNIST/FGSM_train_labels.npy",
        lossObject=keras.losses.CategoricalCrossentropy(),
        epsilon=0.1,
        # n=1024,
        workercount=8,
        chunksize=64,
        constrainer=constrainer,
        force_overwrite=True
    )

    print("\n\n\nMNIST PGD\n")
    AttackDispatcher(
        attack_type="PGD",
        datasetPath="Datasets/MNIST/train_data.npy",
        targetPath="Datasets/MNIST/train_target.npy",
        modelPath="Models/MNIST/maxpool_model.keras",
        adversaryPath="Adversaries/MNIST/PGD_train_data.npy",
        newLabelPath="Adversaries/MNIST/PGD_train_labels.npy",
        lossObject=keras.losses.CategoricalCrossentropy(),
        stepcount=20,
        stepsize=stepsize,
        # n=1024,
        workercount=8,
        chunksize=64,
        constrainer=constrainer,
        force_overwrite=True
    )

    print("\n\n\nMNIST RDSA\n")
    AttackDispatcher(
        attack_type="RDSA",
        datasetPath="Datasets/MNIST/train_data.npy",
        targetPath="Datasets/MNIST/train_target.npy",
        modelPath="Models/MNIST/maxpool_model.keras",
        adversaryPath="Adversaries/MNIST/RDSA_train_data.npy",
        newLabelPath="Adversaries/MNIST/RDSA_train_labels.npy",
        attempts=25,
        categoricalFeatureMaximum=100,
        binCount=100,
        perturbedFeatureCount=100,
        # n=1024,
        workercount=8,
        chunksize=64,
        constrainer=constrainer,
        force_overwrite=True
    )
