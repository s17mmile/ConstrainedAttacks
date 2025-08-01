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
from Evaluation.label_analysis import *

from sklearn.metrics import RocCurveDisplay, roc_auc_score






# Eval script only does one attack at a time but we want to render all the attack resulting histograms for comparison (topodnn)
original_data = np.load("Datasets/TopoDNN/train_data.npy")

constraints = ["spreadLimit", "conserveConstits_spreadLimit", "conserveConstits_spreadLimit_conserveGlobalEnergy", "conserveConstits_spreadLimit_conserveParticleEnergy"]

# Spreadlimit only
for constraint in constraints:
    FGSM_data = np.load(f"Adversaries/TopoDNN/{constraint}/FGSM_train_data.npy")
    PGD_data = np.load(f"Adversaries/TopoDNN/{constraint}/PGD_train_data.npy")
    RDSA_data = np.load(f"Adversaries/TopoDNN/{constraint}/RDSA_train_data.npy")
    
    render_feature_histograms([original_data, FGSM_data, PGD_data, RDSA_data], ["Original", "FGSM", "PGD", "RDSA"], np.arange(0,90), 100, f"Results/TopoDNN/Feature Distributions/{constraint}", "Comparison")