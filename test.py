import os
import numpy as np
import sys
# import keras
# import tensorflow as tf

from Evaluation.dataset_analysis import *
from Helpers.constrainers import *
# data = np.load("Datasets/TopoDNN/train_data.npy", mmap_mode="r")

# sample = data[238].copy()

# print(sample)
# print(np.count_nonzero(sample))

# perturbed = sample + np.random.rand(90)

# print(jetEnergy(sample))
# print(jetEnergy(TopoDNN_spreadLimit(sample)))

# print(jetEnergy(perturbed))
# print(jetEnergy(TopoDNN_conserveConstits(perturbed, sample)))
# print(jetEnergy(TopoDNN_conserveGlobalEnergy(perturbed, sample)))

# fixed = constrainer_TopoDNN_conserveConstits_spreadLimit_conserveGlobalEnergy(perturbed, sample)

# print(jetEnergy(fixed))



data = np.load("Adversaries/TopoDNN/spreadLimit/RDSA_train_data.npy", mmap_mode="r")

