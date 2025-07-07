import numpy as np
import os
import gc
import sys
import multiprocessing
import tqdm
import random

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"

# import tensorflow as tf

import keras
from keras import layers, models
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from itertools import repeat

# Local imports
from Helpers.RDSA_Helpers import GetPDFsAndBinEdges, DistributionShuffleAttackPerVarsParallel
from Helpers.EvaluateModel import getFoolingRatio, getLossAndAccuracy, compute_difference_per_event, get_JSD

