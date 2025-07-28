import numpy as np
import os
import sys
import warnings
import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

import tensorflow as tf
import keras



def RetrainingDispatcher(baseModelPath, retrainingDataPath, trainingTarget, subdivisionCount, epochs):
    '''
        baseModelPath: Path to keras model.
        retrainingDataPath: Path to retraining data (numpy array). Must be compatible with the model's input dimensions.
        trainingTarget. Target labels for the original data. The adversarial attack process is based on the assumption that this target is still accurate.
        subdivisionCount: number of subdivisions for which retraining should be analyzed. E.g. 5 --> Retraining data is split into 5 equal-size subsets on which the model is successively trained.
        epochs: Maximum epoch count for each subdivision training step
    '''

    model = keras.models.load_model(baseModelPath)

    retrainingData = np.load(retrainingDataPath, mmap_mode="r")
    trainingTarget = np.load(retrainingDataPath, mmap_mode="r")
    num_samples = retrainingData.shape[0]

    # Create lists of indices at which retraining data should be split. Includes 0 and num_samples at either end for later indexing.
    subdivisionIndices = [int(i*num_samples/subdivisionCount) for i in range(subdivisionCount+1)]
    print(subdivisionIndices)

    # Callbacks are set to the default callbacks used for other model training in this project, without checkpoints.
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
    ]
    
    # Retrain Model in steps
    for i in range(subdivisionCount):
        lowIndex = subdivisionIndices[i]
        highIndex = subdivisionIndices[i+1]

        data = retrainingData[lowIndex:highIndex]
        target = trainingTarget[lowIndex:highIndex]

        model.fit(
            data,
            target,
            batch_size=128,
            epochs=epochs,
            validation_split=0.15,
            callbacks=callbacks,
        )

        keras.models.save_model(baseModelPath.replace(".keras", f"_retrained_{i}.keras"))

