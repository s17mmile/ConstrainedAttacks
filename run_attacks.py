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



# Arbitrary Constraint for image classifiers
# Add a one-pixel-wide white box (all color channels gets value 1) around any given image.
# Example should be 3d numpy array.
def addBox(example):
    # Failsafe in case I fucked up the shape
    if example.ndim != 3:
        return example

    example[0,:,:] = 1.
    example[-1,:,:] = 1.
    example[:,0,:] = 1.
    example[:,-1,:] = 1.

    return example

# Rescale an arrary linearly from its original range into a given one.
def linearRescale(array, newMin, newMax):
    minimum, maximum = np.min(array), np.max(array)
    m = (newMax - newMin) / (maximum - minimum)
    b = newMin - m * minimum
    scaledArray = m * array + b
    # Remove rounding errors by clipping. The difference is tiny.
    return np.clip(scaledArray, newMin, newMax)

def constrainer_scale_0_1_box(example):
    example = linearRescale(example,0,1)
    example = addBox(example)
    return example

def constrainer_scale_m1_1_box(example):
    example = linearRescale(example,-1,1)
    example = addBox(example)
    return example

def constrainer_TopoDNN_spreadlimit(example):
    # Hardcoded minima and maxima across the training dataset
    min_pT = 0.0
    max_pT = 1.0

    min_eta = -1.909346
    max_eta = 2.119423

    min_phi = -1.588196
    max_phi = 1.443206

    # Constrain pT, eta and phi values by clipping - we might lose some info, but doing a linear rescale here seems like it has more potential to break things than for image classifiers.
    example[0::3] = np.clip(example[0::3], min_pT, max_pT)
    example[1::3] = np.clip(example[1::3], min_eta, max_eta)
    example[2::3] = np.clip(example[2::3], min_phi, max_phi)

    return example

def stepsize(step):
    return 0.05*(1/2**step)



# Selector Panel: Choose which attacks to perform
CIFAR_FGSM = True
CIFAR_PGD = True
CIFAR_RDSA = True

ImageNet_FGSM = True
ImageNet_PGD = True
ImageNet_RDSA = False

MNIST_FGSM = True
MNIST_PGD = True
MNIST_RDSA = True

TopoDNN_FGSM = True
TopoDNN_PGD = True
TopoDNN_RDSA = True



# This extra __main__ specifier is necessary to use multiprocessing without getting a recursion error.
# Unfold each code region to view the case-specific attack configuration.
if __name__ == "__main__":

    if (CIFAR_FGSM):
        try:
            print("\n\n\nCIFAR FGSM\n")
            AttackDispatcher(
                attack_type="FGSM",
                datasetPath="Datasets/CIFAR10/train_data.npy",
                targetPath="Datasets/CIFAR10/train_target.npy",
                modelPath="Models/CIFAR10/base_model.keras",
                adversaryPath="Adversaries/CIFAR10/test/FGSM_train_data.npy",
                originalLabelPath="Adversaries/CIFAR10/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/CIFAR10/test/FGSM_train_labels.npy",
                lossObject=keras.losses.CategoricalCrossentropy(),
                epsilon=0.05,
                return_labels=True,
                n=1,
                workercount=8,
                chunksize=128,
                constrainer=constrainer_scale_0_1_box,
                force_overwrite=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (CIFAR_PGD):
        try:
            print("\n\n\nCIFAR PGD\n")
            AttackDispatcher(
                attack_type="PGD",
                datasetPath="Datasets/CIFAR10/train_data.npy",
                targetPath="Datasets/CIFAR10/train_target.npy",
                modelPath="Models/CIFAR10/base_model.keras",
                adversaryPath="Adversaries/CIFAR10/test/PGD_train_data.npy",
                originalLabelPath="Adversaries/CIFAR10/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/CIFAR10/test/PGD_train_labels.npy",
                lossObject=keras.losses.CategoricalCrossentropy(),
                stepcount=20,
                stepsize=stepsize,
                return_labels=True,
                n=1,
                workercount=8,
                chunksize=128,
                feasibilityProjector=constrainer_scale_0_1_box,
                force_overwrite=True
            )
        except Exception as e:
            print(f"Failure: {e}")
    
    if (CIFAR_RDSA):
        try:   
            print("\n\n\nCIFAR RDSA\n")
            AttackDispatcher(
                attack_type="RDSA",
                datasetPath="Datasets/CIFAR10/train_data.npy",
                targetPath="Datasets/CIFAR10/train_target.npy",
                modelPath="Models/CIFAR10/base_model.keras",
                adversaryPath="Adversaries/CIFAR10/test/RDSA_train_data.npy",
                originalLabelPath="Adversaries/CIFAR10/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/CIFAR10/test/RDSA_train_labels.npy",
                attempts=25,
                categoricalFeatureMaximum=100,
                binCount=100,
                perturbedFeatureCount=300,
                return_labels=True,
                n=1,
                workercount=8,
                chunksize=128,
                constrainer=constrainer_scale_0_1_box,
                force_overwrite=True
            )
        except Exception as e:
            print(f"Failure: {e}")



    if (ImageNet_FGSM):
        try:
            print("\n\n\nImageNet FGSM\n")
            AttackDispatcher(
                attack_type="FGSM",
                datasetPath="Datasets/ImageNet/threshold_data.npy",
                targetPath="Datasets/ImageNet/threshold_target.npy",
                modelPath="Models/ImageNet/base_model.keras",
                adversaryPath="Adversaries/ImageNet/test/FGSM_threshold_data.npy",
                originalLabelPath="Adversaries/ImageNet/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/ImageNet/test/FGSM_threshold_labels.npy",
                lossObject=keras.losses.CategoricalCrossentropy(),
                epsilon=0.05,
                return_labels=True,
                n=1,
                workercount=8,
                chunksize=128,
                constrainer=constrainer_scale_m1_1_box,
                force_overwrite=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (ImageNet_PGD):
        try:
            print("\n\n\nImageNet PGD\n")
            AttackDispatcher(
                attack_type="PGD",
                datasetPath="Datasets/ImageNet/threshold_data.npy",
                targetPath="Datasets/ImageNet/threshold_target.npy",
                modelPath="Models/ImageNet/base_model.keras",
                adversaryPath="Adversaries/ImageNet/test/PGD_threshold_data.npy",
                originalLabelPath="Adversaries/ImageNet/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/ImageNet/test/PGD_threshold_labels.npy",
                lossObject=keras.losses.CategoricalCrossentropy(),
                stepcount=20,
                stepsize=stepsize,
                return_labels=True,
                n=1,
                workercount=8,
                chunksize=128,
                feasibilityProjector=constrainer_scale_m1_1_box,
                force_overwrite=True
            )
        except Exception as e:
            print(f"Failure: {e}")
        
    if (ImageNet_RDSA):
        try:
            print("\n\n\nImageNet RDSA\n")
            AttackDispatcher(
                attack_type="RDSA",
                datasetPath="Datasets/ImageNet/threshold_data.npy",
                targetPath="Datasets/ImageNet/threshold_target.npy",
                modelPath="Models/ImageNet/base_model.keras",
                adversaryPath="Adversaries/ImageNet/test/RDSA_threshold_data.npy",
                originalLabelPath="Adversaries/ImageNet/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/ImageNet/test/RDSA_threshold_labels.npy",
                attempts=25,
                categoricalFeatureMaximum=150,
                binCount=200,
                perturbedFeatureCount=10000,
                return_labels=True,
                n=1,
                workercount=8,
                chunksize=128,
                constrainer=constrainer_scale_m1_1_box,
                force_overwrite=True
            )
        except Exception as e:
            print(f"Failure: {e}")



    if (MNIST_FGSM):
        try:
            print("\n\n\nMNIST FGSM\n")
            AttackDispatcher(
                attack_type="FGSM",
                datasetPath="Datasets/MNIST/train_data.npy",
                targetPath="Datasets/MNIST/train_target.npy",
                modelPath="Models/MNIST/maxpool_model.keras",
                adversaryPath="Adversaries/MNIST/test/FGSM_train_data.npy",
                originalLabelPath="Adversaries/MNIST/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/MNIST/test/FGSM_train_labels.npy",
                lossObject=keras.losses.CategoricalCrossentropy(),
                epsilon=0.05,
                return_labels=True,
                n=1,
                workercount=8,
                chunksize=128,
                constrainer=constrainer_scale_0_1_box,
                force_overwrite=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (MNIST_PGD):
        try:
            print("\n\n\nMNIST PGD\n")
            AttackDispatcher(
                attack_type="PGD",
                datasetPath="Datasets/MNIST/train_data.npy",
                targetPath="Datasets/MNIST/train_target.npy",
                modelPath="Models/MNIST/maxpool_model.keras",
                adversaryPath="Adversaries/MNIST/test/PGD_train_data.npy",
                originalLabelPath="Adversaries/MNIST/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/MNIST/test/PGD_train_labels.npy",
                lossObject=keras.losses.CategoricalCrossentropy(),
                stepcount=20,
                stepsize=stepsize,
                return_labels=True,
                n=1,
                workercount=8,
                chunksize=128,
                feasibilityProjector=constrainer_scale_0_1_box,
                force_overwrite=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (MNIST_RDSA):
        try:
            print("\n\n\nMNIST RDSA\n")
            AttackDispatcher(
                attack_type="RDSA",
                datasetPath="Datasets/MNIST/train_data.npy",
                targetPath="Datasets/MNIST/train_target.npy",
                modelPath="Models/MNIST/maxpool_model.keras",
                adversaryPath="Adversaries/MNIST/test/RDSA_train_data.npy",
                originalLabelPath="Adversaries/MNIST/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/MNIST/test/RDSA_train_labels.npy",
                attempts=25,
                categoricalFeatureMaximum=100,
                binCount=100,
                perturbedFeatureCount=200,
                return_labels=True,
                n=1,
                workercount=8,
                chunksize=128,
                constrainer=constrainer_scale_0_1_box,
                force_overwrite=True
            )
        except Exception as e:
            print(f"Failure: {e}")



    if (TopoDNN_FGSM):
        try:
            print("\n\n\nTopoDNN FGSM\n")
            AttackDispatcher(
                attack_type="FGSM",
                datasetPath="Datasets/TopoDNN/train_data.npy",
                targetPath="Datasets/TopoDNN/train_target.npy",
                modelPath="Models/TopoDNN/base_model.keras",
                adversaryPath="Adversaries/TopoDNN/test/FGSM_train_data.npy",
                originalLabelPath="Adversaries/TopoDNN/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/TopoDNN/test/FGSM_train_labels.npy",
                lossObject=keras.losses.BinaryCrossentropy(),
                epsilon=0.05,
                return_labels=True,
                n=1,
                workercount=8,
                chunksize=128,
                constrainer=constrainer_TopoDNN_spreadlimit,
                force_overwrite=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (TopoDNN_PGD):
        try:
            print("\n\n\nTopoDNN PGD\n")
            AttackDispatcher(
                attack_type="PGD",
                datasetPath="Datasets/TopoDNN/train_data.npy",
                targetPath="Datasets/TopoDNN/train_target.npy",
                modelPath="Models/TopoDNN/base_model.keras",
                adversaryPath="Adversaries/TopoDNN/test/PGD_train_data.npy",
                originalLabelPath="Adversaries/TopoDNN/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/TopoDNN/test/PGD_train_labels.npy",
                lossObject=keras.losses.BinaryCrossentropy(),
                stepcount=20,
                stepsize=stepsize,
                return_labels=True,
                n=1,
                workercount=8,
                chunksize=128,
                feasibilityProjector=constrainer_TopoDNN_spreadlimit,
                force_overwrite=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (TopoDNN_RDSA):
        try:
            print("\n\n\nTopoDNN RDSA\n")
            AttackDispatcher(
                attack_type="RDSA",
                datasetPath="Datasets/TopoDNN/train_data.npy",
                targetPath="Datasets/TopoDNN/train_target.npy",
                modelPath="Models/TopoDNN/base_model.keras",
                adversaryPath="Adversaries/TopoDNN/test/RDSA_train_data.npy",
                originalLabelPath="Adversaries/TopoDNN/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/TopoDNN/test/RDSA_train_labels.npy",
                attempts=25,
                categoricalFeatureMaximum=100000,
                binCount=1000,
                perturbedFeatureCount=15,
                return_labels=True,
                n=1,
                workercount=8,
                chunksize=128,
                constrainer=constrainer_TopoDNN_spreadlimit,
                force_overwrite=True
            )
        except Exception as e:
            print(f"Failure: {e}")