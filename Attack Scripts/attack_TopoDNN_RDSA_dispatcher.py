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




# TODO write actual constrainer 
def constrainer(example):
    return example



# This extra specifier is necessary to use multiprocessing without getting a recursion error.
if __name__ == "__main__":

    # (All values that aren't set by the preprocessing step are continuous for TopoDNN's data.)
    AttackDispatcher(
        attack_type="RDSA",
        datasetPath="Datasets/TopoDNN/train_data.npy",
        targetPath="Datasets/TopoDNN/train_target.npy",
        modelPath="Models/TopoDNN/base_model.keras",
        adversaryPath="Adversaries/TopoDNN/RDSA_train_data_10.npy",
        newLabelPath="Adversaries/TopoDNN/RDSA_train_labels_10.npy",
        attempts=25,
        categoricalFeatureMaximum=100000,
        binCount=1000,
        perturbedFeatureCount=15,
        n=1024,
        workercount=8,
        chunksize=64,
        constrainer=constrainer,
        force_overwrite=False
    )