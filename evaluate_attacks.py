import numpy as np
import os
import sys
import warnings
import tqdm

sys.path.append(os.getcwd())

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

import tensorflow as tf
import keras



from Evaluation.evaluation_dispatch import EvaluationDispatcher 

# Selector panel

# CIFAR, ImageNet and MNIST will run (unless otherwise specified) using the scaled data - no further constraints.
CIFAR_FGSM = True
CIFAR_PGD = True
CIFAR_RDSA = True

ImageNet_FGSM = True
ImageNet_PGD = True
ImageNet_RDSA = True

MNIST_FGSM = True
MNIST_PGD = True
MNIST_RDSA = True



# TopoDNN has more options. These can be seen in run_attacks.py with a bit more explanation.
TopoDNN_FGSM_clip = True
TopoDNN_FGSM_constits_clip = True
TopoDNN_FGSM_constits_clip_globalEnergy = True

TopoDNN_PGD_clip = True
TopoDNN_PGD_constits_clip = True
TopoDNN_PGD_constits_clip_globalEnergy = True

TopoDNN_RDSA_clip = True
TopoDNN_RDSA_constits_clip = True
TopoDNN_RDSA_constits_clip_globalEnergy = True



if __name__ == "__main__":

    # Run eval with all the different configs
    try:
        EvaluationDispatcher(
            originalDatasetPath="Datasets/CIFAR10/train_data.npy",
            perturbedDatasetPath="Adversaries/CIFAR10_FGSM_train_data.npy",
            originalTargetPath="Datasets/CIFAR10/train_target.npy",
            testDataPath="Datasets/CIFAR10/test_data.npy",
            testTargetPath="Datasets/CIFAR10/test_target.npy",
            baseModelPath="Models/CIFAR10/base_model.keras",
            retrainedModelPaths=[
                "Models/CIFAR10/Retrained/XYZ.keras"
                ... TODO
            ],
            histogramFeatures = [(15,15,0),(15,,15,1),(15,15,2)],
            attackName = "\"FGSM, ranged\"",
            resultDirectory="Results/CIFAR10/",
            computeCorrelation=True
        )
    except Exception as e:
        print(f"Failure: {e}")