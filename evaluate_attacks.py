import numpy as np
import os
import sys
import tqdm

# Catch unnecessary sklearn warnings. Not super clean.
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

sys.path.append(os.getcwd())

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

import tensorflow as tf
import keras



from Evaluation.evaluation_dispatch import EvaluationDispatcher 

# Hard-Coded number of retraining subdivions
retraining_subdivisions = 5

# Selector panel
# CIFAR, ImageNet and MNIST will run (unless otherwise specified) using the scaled data - no further constraints.
CIFAR_FGSM_scaled = True
CIFAR_PGD_scaled = True
CIFAR_RDSA_scaled = True

CIFAR_FGSM_scaled_boxed = True
CIFAR_PGD_scaled_boxed = True
CIFAR_RDSA_scaled_boxed = True



ImageNet_FGSM_scaled = True
ImageNet_PGD_scaled = True
ImageNet_RDSA_scaled = True

ImageNet_FGSM_scaled_boxed = True
ImageNet_PGD_scaled_boxed = True
ImageNet_RDSA_scaled_boxed = True



MNIST_FGSM_scaled = True
MNIST_PGD_scaled = True
MNIST_RDSA_scaled = True

MNIST_FGSM_scaled_boxed = True
MNIST_PGD_scaled_boxed = True
MNIST_RDSA_scaled_boxed = True



# TopoDNN has more options. These can be seen in run_attacks.py with a bit more explanation.
TopoDNN_FGSM_clip = True
TopoDNN_PGD_clip = True
TopoDNN_RDSA_clip = True

TopoDNN_FGSM_constits_clip = True
TopoDNN_RDSA_constits_clip = True
TopoDNN_PGD_constits_clip = True

TopoDNN_FGSM_constits_clip_globalEnergy = True
TopoDNN_PGD_constits_clip_globalEnergy = True
TopoDNN_RDSA_constits_clip_globalEnergy = True

TopoDNN_FGSM_constits_clip_particleEnergy = True
TopoDNN_PGD_constits_clip_particleEnergy = True
TopoDNN_RDSA_constits_clip_particleEnergy = True

if __name__ == "__main__":

    # Run eval with all the different configs
    if (CIFAR_FGSM_scaled):
        # try:
        baseModelPath="Models/CIFAR10/base_model.keras"
        attackName = "FGSM_scaled"
        retrainedModelPaths = [os.path.join(os.path.dirname(baseModelPath), attackName, os.path.basename(baseModelPath).replace(".keras", f"_retrained_{i}.keras")) for i in range(retraining_subdivisions)]

        print(retrainedModelPaths)

        EvaluationDispatcher(
            originalDatasetPath="Datasets/CIFAR10/train_data.npy",
            perturbedDatasetPath="Adversaries/CIFAR10/scaled/FGSM_train_data.npy",
            originalTargetPath="Datasets/CIFAR10/train_target.npy",
            testDataPath="Datasets/CIFAR10/test_data.npy",
            testTargetPath="Datasets/CIFAR10/test_target.npy",
            baseModelPath="Models/CIFAR10/base_model.keras",
            retrainedModelPaths=retrainedModelPaths,
            histogramFeatures = [(15,15,0),(15,15,1),(15,15,2)],
            attackName = attackName,
            resultDirectory="Results/CIFAR10/",
            computeCorrelation=True
        )
        # except Exception as e:
        #     print(f"Failure: {e}")