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


def RetrainingDispatcher(baseModelPath, retrainingDataPath, trainingTargetPath, subdivisionCount, epochs, attackName):
    '''
        baseModelPath: Path to keras model.
        retrainingDataPath: Path to retraining data (numpy array). Must be compatible with the model's input dimensions.
        trainingTarget. Target labels for the original data. The adversarial attack process is based on the assumption that this target is still accurate.
        subdivisionCount: number of subdivisions for which retraining should be analyzed. E.g. 5 --> Retraining data is split into 5 equal-size subsets on which the model is successively trained.
        epochs: Maximum epoch count for each subdivision training step
        attackName: name of attack. Will be used in file name
    '''

    print(f"\n\n\nRetraining\n{baseModelPath}\nusing data from\n{retrainingDataPath}.\n")

    model = keras.models.load_model(baseModelPath, compile = True)

    retrainingData = np.load(retrainingDataPath, mmap_mode="r")
    trainingTarget = np.load(trainingTargetPath, mmap_mode="r")
    num_samples = retrainingData.shape[0]

    # Create lists of indices at which retraining data should be split. Includes 0 and num_samples at either end for later indexing.
    subdivisionIndices = [int(i*num_samples/subdivisionCount) for i in range(subdivisionCount+1)]

    try:
        os.makedirs(os.path.join(os.path.dirname(baseModelPath), attackName))
    except:
        print("Target dir exists, skipping ceration. Will overwrite potentially existing files.")

    # Retrain Model in steps
    for i in range(subdivisionCount):

        # Callbacks are set to only save the best model (lowest loss) so far to disk. I wanted to do highest accuracy, but that would rely on that metric always being available.
        # Stop early if val loss doesn't decrease three times in a row.

        filepath = os.path.join(os.path.dirname(baseModelPath), attackName, os.path.basename(baseModelPath).replace(".keras", f"_retrained_{i}.keras"))
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
            keras.callbacks.ModelCheckpoint(
                filepath = filepath,
                monitor='val_loss',
                mode='min',
                save_best_only=True
            )
        ]

        # Extract the correct data slice
        lowIndex = subdivisionIndices[i]
        highIndex = subdivisionIndices[i+1]

        data = retrainingData[lowIndex:highIndex]
        target = trainingTarget[lowIndex:highIndex]

        # To ensure the validation data doesn't just contain one class (as our datasets are typically grouped by class), we shuffle.
        # Data and target must of course be shuffled the same way, so we use a permutation.
            # In the case of TopoDNN, using the extra validation dataset might work, but since it's not been attacked it would be incosistent with other training.
            # However, it may arguably be a more useful validation tool *because* it's not been attacked, thus better representing the rules we want.
            # Dunno, that's something to discuss.
        permutation = np.random.permutation(data.shape[0])
        data = data[permutation]
        target = target[permutation]

        # Continue training
        print(f"Retraining with data subdivision #{i}:")

        model.fit(
            data,
            target,
            batch_size=100,
            epochs=epochs,
            validation_split=0.15,
            callbacks=callbacks,
        )

    print()