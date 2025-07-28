import numpy as np
import os
import sys

sys.path.append(os.getcwd())

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

import tensorflow as tf
import keras

from Evaluation.dataset_analysis import *
from Helpers.constrainers import *




model = keras.models.load_model("Models/CIFAR10/base_model.keras")
data = np.load("Datasets/CIFAR10/train_data.npy", mmap_mode="r")

print(model.loss())