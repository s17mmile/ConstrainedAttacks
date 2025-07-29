import numpy as np
import os
import sys
import warnings
import tqdm

sys.path.append(os.getcwd())

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

import tensorflow as tf
import keras

from Helpers.constrainers import *



# FGSM and RDSA attacks do not need to be fully re-run to test different constraints, as the constraint is just tacked on to the end.
# As such, we can just apply the constraint to a run of the data that doesn't have the constraint yet.
# PGD is different when using a constrainer as a feasibilityProjector, as the constraint comes into effect throughout the computation.
if __name__ == "__main__":
    # Arbitrary constraints on image-based classifiers



    # TopoDNN extra constraints - the "Base constraint" is just the range constraint (spreadLimit).

    # Load original data. The read-only mmap might actually cost us a tiny bit of performance (not important here) but is a nice way to make sure we don't break the original dataset. 
    original_data = np.load("Datasets/TopoDNN/train_data.npy", allow_pickle=True, mmap_mode="r")

    # Load, constrain (conserveConstits), store and delete.
    # (We will need to load several times as the ApplyToAll function works in-place (consequence of numpy passing arrays by ref).)
    FGSM_data = np.load("Adversaries/TopoDNN/spreadLimit/FGSM_train_data.npy", mmap_mode=None)
    TopoDNN_applyToAll(FGSM_data, constrainer_TopoDNN_conserveConstits_spreadLimit, original_data)
    np.save("Adversaries/TopoDNN/conserveConstits_spreadLimit/FGSM_train_data.npy", FGSM_data)
    del FGSM_data

    # Load, constrain (conserveConstits, globalEnergy), store and delete.
    FGSM_data = np.load("Adversaries/TopoDNN/spreadLimit/FGSM_train_data.npy", mmap_mode=None)
    TopoDNN_applyToAll(FGSM_data, constrainer_TopoDNN_conserveConstits_spreadLimit_conserveGlobalEnergy, original_data)
    np.save("Adversaries/TopoDNN/conserveConstits_spreadLimit_gconserveGlobalEnergy/FGSM_train_data.npy", FGSM_data)
    del FGSM_data

    # Load, constrain (conserveConstits, particleEnergy), store and delete.
    FGSM_data = np.load("Adversaries/TopoDNN/spreadLimit/FGSM_train_data.npy", mmap_mode=None)
    TopoDNN_applyToAll(FGSM_data, constrainer_TopoDNN_conserveConstits_spreadLimit_conserveParticleEnergy, original_data)
    np.save("Adversaries/TopoDNN/conserveConstits_spreadLimit_gconserveParticleEnergy/FGSM_train_data.npy", FGSM_data)
    del FGSM_data



    # Now, do the exact same thing for RDSA :)
    # Load, constrain (conserveConstits), store and delete.
    # (We will need to load several times as the ApplyToAll function works in-place (consequence of numpy passing arrays by ref).)
    RDSA_data = np.load("Adversaries/TopoDNN/spreadLimit/RDSA_train_data.npy", mmap_mode=None)
    TopoDNN_applyToAll(RDSA_data, constrainer_TopoDNN_conserveConstits_spreadLimit, original_data)
    np.save("Adversaries/TopoDNN/conserveConstits_spreadLimit/RDSA_train_data.npy", RDSA_data)
    del RDSA_data

    # Load, constrain (conserveConstits, globalEnergy), store and delete.
    RDSA_data = np.load("Adversaries/TopoDNN/spreadLimit/RDSA_train_data.npy", mmap_mode=None)
    TopoDNN_applyToAll(RDSA_data, constrainer_TopoDNN_conserveConstits_spreadLimit_conserveGlobalEnergy, original_data)
    np.save("Adversaries/TopoDNN/conserveConstits_spreadLimit_gconserveGlobalEnergy/RDSA_train_data.npy", RDSA_data)
    del RDSA_data

    # Load, constrain (conserveConstits, particleEnergy), store and delete.
    RDSA_data = np.load("Adversaries/TopoDNN/spreadLimit/RDSA_train_data.npy", mmap_mode=None)
    TopoDNN_applyToAll(RDSA_data, constrainer_TopoDNN_conserveConstits_spreadLimit_conserveParticleEnergy, original_data)
    np.save("Adversaries/TopoDNN/conserveConstits_spreadLimit_gconserveParticleEnergy/RDSA_train_data.npy", RDSA_data)
    del RDSA_data



    del original_data

    # RDSA_base_data = np.load("Adversaries/TopoDNN/spreadLimit/RDSA_train_data.npy", mmap_mode="r")