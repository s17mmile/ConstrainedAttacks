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

example = np.array([0,1,2,3,4,5,6,7,8,9])

adversary = tf.convert_to_tensor(example)

example[:] += 2

print(example)
print(adversary)

adversary = adversary + 1

print(example)
print(adversary)