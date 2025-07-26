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
CIFAR_FGSM = True
CIFAR_FGSM = True
CIFAR_FGSM = True

ImageNet_FGSM = True
ImageNet_FGSM = True
ImageNet_FGSM = True

CIFAR_FGSM = True
CIFAR_FGSM = True
CIFAR_FGSM = True

CIFAR_FGSM = True
CIFAR_FGSM = True
CIFAR_FGSM = True


if __name__ == "__main__":

    # Run eval with all the different configs
