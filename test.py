import os
import numpy as np
import sys
# import keras
# import tensorflow as tf

from Evaluation.dataset_analysis import *

array1 = np.load("Datasets/CIFAR-10/train_data.npy", mmap_mode="r")
array2 = np.load("Datasets/CIFAR-10/FGSM_train_data.npy", mmap_mode="r")

# render_feature_histograms([array1, array2], ["CIFAR_base", "CIFAR_FGSM"], [(0,0,0)], 100, "Results/CIFAR10/Histograms")

render_correlation_matrix(array1[:,:,:], "Results/CIFAR10/Correlations/unmodified.png")