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
def constrained_FGSM(model, example, target, lossObject, epsilon = 0.1, constrainer = None, return_labels = False):
    '''
        Performs FGSM attack on a single example with a given constraining function (if given).

        Params:
            model: a pre-trained keras model
            example: singular model input. Can be multidimensional.
            target: the correct classification label for the given instance (one-hot).
            lossObject: loss function to be used, quantifying the difference between the prediction and correct label. Typically supplied by tensorflow or keras.
            epsilon: perturbation scaler, constant.
            constrainer: a function which takes in and returns an example as given here, and performs some projection/transformation operation to ensure case-specific feasibility.
            return_labels: return_labels: simple boolean that governs whether or not the labels the model assigns (to the original AND perturbed samples) are returned.

        Returns: Adversary and, optionally, the label associated with it.
    '''

    # Data formatting for use with GradientTape and as a model input
    # --> Within this function, manipulate everything as tensors. Only transform back to numpy at the end.
    adversary = tf.convert_to_tensor([example])

    with tf.GradientTape() as tape:
        tape.watch(adversary)
        prediction = model(adversary, training = False)[0]
        loss = lossObject(target, prediction)

        # Get the gradients of the loss w.r.t to the input example.
        gradient = tape.gradient(loss, adversary)
        # Get the sign of the gradients to create the perturbation
        gradient_sign = tf.sign(gradient)
        # Apply perturbation
        adversary = adversary + epsilon * gradient_sign

    # Convert adversary to numpy. A user whould be able to apply custom constrainers to numpy arrays and not have to work with tensors.
    adversary = adversary.numpy()[0]

    # If given: apply the final constrainer.
    # This can result in a decrease of the loss function, there's not really any way to avoid that.
    # We essentially hope that adding the constraint doesn't fix the prediction.
    if constrainer is not None:
        adversary = constrainer(adversary, example)

    # Compte and return the labels if wanted
    if (return_labels):
        originalLabel = model(tf.convert_to_tensor([example]), training = False).numpy()[0]
        newLabel = model(tf.convert_to_tensor([adversary]), training = False).numpy()[0]
        return adversary, originalLabel, newLabel
    else:
        return adversary



# Runs the constrained_FGSM function in parallel using a starmap 
def parallel_constrained_FGSM(model, dataset, targets, lossObject, epsilon = 0.1, constrainer = None, return_labels = False, workercount = 1, chunksize = 1):
    '''
        Performs constrained FGSM attack on a whole set of examples with a given constraining function (if given).
        The use of tqdm means that a progress bar will indicate progress during computation.

        Params:
            model: pre-trained keras model
            dataset: Set of model inputs, tested using numpy arrays. Similar indexable structures such as tensors may work too.
            targets: the correct classification labels for each instance, one-hot.
            lossObject: loss function to be used, quantifying the difference between the prediction and correct label. Typically supplied by tensorflow or keras.
            epsilon: perturbation scaler - constant for now. Might modify to iterate towards smallest sufficient modification.
            constrainer: a function which takes in and returns an example as given here, and performs some projection operation to ensure case-specific feasibility. Optional.
            return_labels: return_labels: simple boolean that governs whether or not the labels the model assigns (to the original AND perturbed samples) are returned.
            
            workercount: How many threads should run in parallel. Recommended to be about half of the running device's thread count. Optional.
            chunksize: chunk size used for the starmap call. Approximately the number of examples assigned to each workrer at a time. Optional.

        Returns:
            Adversaries (large numpy array)
            Optionally, the labels associated with the original and adversarial samples.
    '''

    with multiprocessing.get_context("spawn").Pool(workercount) as p:
        results = p.starmap(constrained_FGSM, tqdm.tqdm(zip(
                                repeat(model), dataset, targets, repeat(lossObject), repeat(epsilon), repeat(constrainer)
                            ),
                            total = dataset.shape[0]), chunksize=chunksize)
    


    # Format data for output
    if return_labels:
        adversaries = np.array([event[0] for event in results])
        originalLabels = np.array([event[1] for event in results])
        adversarialLabels = np.array([event[2] for event in results])
        return adversaries, originalLabels, adversarialLabels
    else:
        return np.array(results)
