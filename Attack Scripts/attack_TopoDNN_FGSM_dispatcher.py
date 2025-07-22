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

    AttackDispatcher(
        attack_type="FGSM",
        datasetPath="Datasets/TopoDNN/train_data.npy",
        targetPath="Datasets/TopoDNN/train_target.npy",
        modelPath="Models/TopoDNN/base_model.keras",
        adversaryPath="Adversaries/TopoDNN/FGSM_train_data.npy",
        newLabelPath="Adversaries/TopoDNN/FGSM_train_labels.npy",
        lossObject=keras.losses.BinaryCrossentropy(),
        epsilon=0.1,
        n=1024*8,
        workercount=8,
        chunksize=1024,
        constrainer=constrainer,
        force_overwrite=False
    )

