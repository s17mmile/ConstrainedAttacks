import numpy as np
import os
import gc
import sys
import multiprocessing
import tqdm
import random

sys.path.append(os.getcwd())

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import tensorflow as tf
import keras

# We're ONLY testing the TopoDNN models that don't have any extra rpeprocessing steps (check the training script in the topodnn repo for context).
# This can be checked in each model's .json file. 
modelnames = ["topodnnmodel","topodnnmodel_30","topodnnmodel_v1","topodnnmodel_v2","topodnnmodel_v3","topodnnmodel_v4"]

test_data = np.load("Datasets/TopoDNN/test_data.npy", allow_pickle=True)
test_target = np.load("Datasets/TopoDNN/test_target.npy", allow_pickle=True)

for modelname in modelnames:
    model = keras.models.load_model(f"Models/TopoDNN/{modelname}.keras")

    model.evaluate(test_data, test_target, verbose=1)