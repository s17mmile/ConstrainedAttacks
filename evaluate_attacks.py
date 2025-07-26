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
