import numpy as np
import os
import gc
import sys
import multiprocessing
import tqdm
import random

from scipy import stats
from itertools import repeat

import tensorflow as tf
import keras



# Generate an adversarial example from a given (unperturbed) example.
# The epsilon and constrainer are not set by this definition but given default values, making them optional arguments.
def constrained_FGSM(model, example, target, lossObject, epsilon = 0.01, constrainer = None):
    '''
        Performs FGSM attack on a single example with a given constraining function (if given).

        Params:
            model: a pre-trained keras model
            example: singular model input.
            target: the correct classification label for the given instance (one-hot).
            lossObject: loss function to be used, quantifying the difference between the prediction and correct label. Typically supplied by tensorflow or keras - unsure of exact format.
            epsilon: perturbation scaler - constant for now. Might modify to iterate towards smallest sufficient modification.
            constrainer: a function which takes in and returns an example as given here, and performs some projection operation to ensure case-specific feasibility. Optional.

        Returns: Adversary (1D numpy array) and the label associated with it.
    '''

    # Data formatting for use with GradientTape and as a model input
    # --> Within this function, manipulate everything as tensors. Only transform back to numpy at the end.
    exampleTensor = tf.convert_to_tensor(np.array([example]))

    with tf.GradientTape() as tape:
        tape.watch(exampleTensor)
        prediction = model(exampleTensor)[0]
        loss = lossObject(target, prediction)

    # Get the gradients of the loss w.r.t to the input example.
    gradient = tape.gradient(loss, exampleTensor)
    # Get the sign of the gradients to create the perturbation
    gradient_sign = tf.sign(gradient)

    adversary = exampleTensor + epsilon * gradient_sign

    if constrainer is not None:
        adversary = constrainer(adversary)

    newLabel = model(adversary, training = False)

    # Convert back to numpy
    adversary = adversary.numpy()[0]
    newLabel = newLabel.numpy()[0]

    return adversary, newLabel



# Runs the constrained_FGSM function in parallel using a starmap 
def parallel_constrained_FGSM(model, dataset, targets, lossObject, epsilon = 0.1, constrainer = None, workercount = 1, chunksize = 1):
    '''
        Performs constrained FGSM attack on a whole set of examples with a given constraining function (if given).
        The use of tqdm means that a progress bar will indicate progress during computation.

        Params:
            model: pre-trained keras model
            dataset: Set of model inputs. 2D numpy array.
            targets: the correct classification labels for each instance. 2D numpy array.
            lossObject: loss function to be used, quantifying the difference between the prediction and correct label. Typically supplied by tensorflow or keras - unsure of exact format.
            epsilon: perturbation scaler - constant for now. Might modify to iterate towards smallest sufficient modification.
            constrainer: a function which takes in and returns an example as given here, and performs some projection operation to ensure case-specific feasibility. Optional.

            workercount: How many threads should run in parallel. Recommended to be about half of the running device's thread count. Optional.
            chunksize: chunk size used for the starmap call. Approximately the number of examples assigned to each workrer at a time. Optional.

        Returns two lists:
            Adversaries (numpy arrays)
            The labels associated with them
    '''

    # Return value contains a list for each perturbed example, with:
        # - the perturbed data
        # - the new label given to this data by the model. If this is different from the original, it's a success

    adversaries = []
    newLabels = []

    with multiprocessing.get_context("spawn").Pool(workercount) as p:
        results = p.starmap(constrained_FGSM, tqdm.tqdm(zip(
            repeat(model), dataset, targets, repeat(lossObject), repeat(epsilon), repeat(constrainer)),
            total = dataset.shape[0]), chunksize=chunksize)
    
    # Format data for output
    print("Formatting results...")
    for event in results:
        adversaries.append(event[0])
        newLabels.append(event[1])

    return np.array(adversaries), np.array(newLabels)
