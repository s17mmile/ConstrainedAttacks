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
# The stepsize, stepcount, feasaibilityProjector and constrainer are not set by this definition but given default values, making them optional arguments.
# --> This is an improved, iterated version of the FGSM attack. Performing this with stepsize = epsilon, stepcount = 1 and feasaibilityProjector = None should yield exactly the same results.
# --> The feasibilityProjector and final constrainer are given the option to be different, though in some cases it might make sense for the final projection to be the same feasibility constraint as before.
# In this case, no constrainer argument is required.
def constrained_PGD(model, example, target, lossObject, stepcount = 10, stepsize = 0.01, feasibilityProjector = None, constrainer = None):
    '''
        Performs PGD attack on a single example with a given constraining function (if given). If not feasibilityProjector is given, this is just iuterated gradient descent.

        Params:
            model: a pre-trained keras model
            example: singular model input. Should be a simple 1D numpy array. Pass by value!
            target: the correct classification label for the given instance.
            lossObject: loss function to be used, quantifying the difference between the prediction and correct label. Typically supplied by tensorflow or keras - unsure of exact format.
            stepcount: number of gradient descent and projection cycles that should be performed.
            stepsize: perturbation scaler - constant for now. Scales the gradient (sign?) by this amount for each gradient descent step.
            feasibilityProjector: function that takes in an example (np array) and projects it to a case-specific feasibility region. Typically, this is an orthogonal projection onto a linearly defined subspace.
            constrainer: a function which takes in and returns an example as given here, and performs some projection operation to ensure case-specific feasibility. Optional.

        Returns: Adversary (1D numpy array) and the label associated with it.
    '''

    # Data formatting for use with GradientTape and as a model input
    # --> Within this function, manipulate everything as tensors. Only transform back to numpy at the end.
    # print("convert 1")
    adversary = tf.convert_to_tensor(np.array([example]))

    # We re-instantiate the GradientTape each time. I'm not sure if this is a preformance loss, but it feels like it.
    # I tried the persistent tape, but could not get it to work.
    for step in range(stepcount):
        # print("step",step)
        with tf.GradientTape(persistent = True) as tape:
            tape.watch(adversary)
            # Run model on current adversary
            prediction = model(adversary, training = False)[0]

            # Stop early if the prediction is already incorrect.
            if np.argmax(prediction) != np.argmax(target):
                break

            # Calculate loss of current adversary prediction
            loss = lossObject(target, prediction)
        
        # Get the gradients of the loss w.r.t to the current adversary.
        gradient = tape.gradient(loss, adversary)
        # Get the sign of the gradients to create the perturbation
        gradient_sign = tf.sign(gradient)
        # Apply Gradient perturbation
        adversary = adversary + stepsize(step) * gradient_sign

        # If given: apply the feasibility projector.
        if feasibilityProjector is not None:
            adversary = adversary.numpy()[0]
            adversary = feasibilityProjector(adversary)
            adversary = tf.convert_to_tensor([adversary])

    # If given: apply the final constrainer.
    if constrainer is not None:
        adversary = adversary.numpy()[0]
        adversary = constrainer(adversary)
        adversary = tf.convert_to_tensor([adversary])

    newLabel = model(adversary, training = False)

    # Convert back to numpy
    adversary = adversary.numpy()[0]
    newLabel = newLabel.numpy()[0]

    return adversary, newLabel



# Runs the constrained_FGSM function in parallel using a starmap 
def parallel_constrained_PGD(model, dataset, targets, lossObject, stepcount = 10, stepsize = 0.01, feasibilityProjector = None, constrainer = None, workercount = 1, chunksize = 1):
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
        results = p.starmap(constrained_PGD, tqdm.tqdm(zip(
            repeat(model), dataset, targets, repeat(lossObject), repeat(stepcount), repeat(stepsize), repeat(feasibilityProjector), repeat(constrainer)),
            total = dataset.shape[0]), chunksize=chunksize)
    
    # Format data for output
    print("Formatting results...")
    for event in results:
        adversaries.append(event[0])
        newLabels.append(event[1])

    return np.array(adversaries), np.array(newLabels)
