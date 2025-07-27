import os
import numpy as np
import sys
# import keras
# import tensorflow as tf

from Evaluation.dataset_analysis import *

array1 = np.load("Datasets/TopoDNN/train_data.npy", mmap_mode="r")

render_feature_histograms([array1], ["TopoDNN_base"], np.arange(90), 100, "Results/TopoDNN/Feature Distributions", "TopoDNN_base")