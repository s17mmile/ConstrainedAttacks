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

# def test(data):
#     check = np.isnan(data)
#     print(np.unique(check,return_counts=True))
#     print(np.min(data))
#     print(np.unravel_index(np.argmin(data), data.shape))
#     print(np.max(data))
#     print(np.unravel_index(np.argmax(data), data.shape))
#     print()

# def num_constits(jet):
#     count = 30
#     tol = 1e-12
#     for i in range(0,90,3):
#         if abs(jet[i]) < tol and abs(jet[i+1]) < tol and abs(jet[i+2]) < tol:
#             count -= 1
#     return count

# def extract_pT(jet):
#     return jet[0::3]

# data = np.load("Datasets/TopoDNN/train_data.npy", mmap_mode="r")
# # test(data)

# # adv1 = np.load("Adversaries/TopoDNN/conserveConstits_spreadLimit/PGD_train_data.npy", mmap_mode="r")
# # test(adv1)

# adv2 = np.load("Adversaries/TopoDNN/conserveConstits_spreadLimit_conserveGlobalEnergy/PGD_train_data.npy", mmap_mode="r")
# # test(adv2)

# adv3 = np.load("Adversaries/TopoDNN/conserveConstits_spreadLimit_conserveParticleEnergy/PGD_train_data.npy", mmap_mode="r")
# # test(adv3)

# for i in range(1024):
#     print(i)
#     print(num_constits(data[i]))
#     print(num_constits(adv2[i]))
#     print(num_constits(adv3[i]))