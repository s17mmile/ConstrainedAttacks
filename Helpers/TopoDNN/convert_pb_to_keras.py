# Important: this script is not actually in use but was necessary in order to convert the pre-trained TopoDNN models into a usable .keras format.
# To do this, old versions of both keras and tensorflow are necessary as the old folder-based model structure is deprecated. This uses keras and tensorflow 2.15.0 for conversion.
# Afterwards, I reinstalled tf 2.16.0 and keras 3.9.2 for the rest of the work, so ignore this script :) 

# Also, not all of these models are actually used as some of them are based on extra preprocessing steps that are not done in this project. This is just for completeness.

import os
import sys
import numpy as np

sys.path.append(os.getcwd())

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

model_names =   [
                    "topodnnmodel",
                    "topodnnmodel_30",
                    "topodnnmodel_pt",
                    "topodnnmodel_pt0",
                    "topodnnmodel_standardize_pt",
                    "topodnnmodel_v1",
                    "topodnnmodel_v2",
                    "topodnnmodel_v3",
                    "topodnnmodel_v4"  
                ]


for name in model_names:
    model = keras.models.load_model("Models/TopoDNN/"+name)
    model.summary()
    model.save("Models/TopoDNN/"+name+".keras")