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
def constrained_PGD(model, example, target, lossObject, stepcount = 10, stepsize = 0.01, feasibilityProjector = None, constrainer = None, return_labels = False):
    '''
        Performs PGD attack on a single example with a given constraining function (if given). If not feasibilityProjector is given, this is just iuterated gradient descent.

        Params:
            model: a pre-trained keras model
            example: singular model input.
            target: the correct classification label for the given instance. One-hot.
            lossObject: loss function to be used, quantifying the difference between the prediction and correct label. Typically supplied by tensorflow or keras - unsure of exact format.
            stepcount: number of gradient descent and projection cycles that should be performed.
            stepsize: Callable function that takes in the step number and returns a step size to allow variability. Scales the gradient sign by this amount for each gradient descent step.
            feasibilityProjector: function that takes in an example (np array) and projects it to a case-specific feasibility region. Typically, this is an orthogonal projection onto a linearly defined subspace.
            constrainer: a function which takes in and returns an example as given here, and performs some projection operation to ensure case-specific feasibility. Optional.
            return_labels: return_labels: simple boolean that governs whether or not the labels the model assigns (to the original AND perturbed samples) are returned.

        Returns: Adversary and, optionally, the label associated with it and the original sample.
    '''

    # Data formatting for use with GradientTape and as a model input
    # --> Within this function, manipulate everything as tensors. Only transform back to numpy at the end.
    adversary = tf.convert_to_tensor(np.array([example]))

    # If given: apply the feasibility projector. We do this once BEFORE the PGD attack to avoid outlier values from slipping through by trigegring the early stop condition.
    if feasibilityProjector is not None:
        adversary = adversary.numpy()[0]
        adversary = feasibilityProjector(adversary, example)
        adversary = tf.convert_to_tensor([adversary])



    # We re-instantiate the GradientTape each time. I'm not sure if this is a performance loss, but it very well might be.
    # I tried the persistent tape, but could not get it to work.

    for step in range(stepcount):
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
            adversary = feasibilityProjector(adversary, example)
            adversary = tf.convert_to_tensor([adversary])

    # Convert adversary to numpy. A user whould be able to apply custom constrainers to numpy arrays and not have to work with tensors.
    adversary = adversary.numpy()[0]

    # If given: apply the final constrainer.
    # This can result in a decrease of the loss function, there's not really any way to avoid that.
    # We essentially hope that adding the constraint doesn't fix the prediction.
    if constrainer is not None:
        adversary = constrainer(adversary, example)

    # Compute and return the labels if wanted
    if (return_labels):
        originalLabel = model(tf.convert_to_tensor([example]), training = False).numpy()[0]
        newLabel = model(tf.convert_to_tensor([adversary]), training = False).numpy()[0]
        return adversary, originalLabel, newLabel
    else:
        return adversary



# Runs the constrained_FGSM function in parallel using a starmap 
def parallel_constrained_PGD(model, dataset, targets, lossObject, stepcount = 10, stepsize = 0.01, feasibilityProjector = None, constrainer = None, return_labels = False, workercount = 1, chunksize = 1):
    '''
        Performs constrained FGSM attack on a whole set of examples with a given constraining function (if given).
        The use of tqdm means that a progress bar will indicate progress during computation.

        Params:
            model: a pre-trained keras model
            example: singular model input. Should be a simple 1D numpy array. Pass by value!
            target: the correct classification label for the given instance.
            lossObject: loss function to be used, quantifying the difference between the prediction and correct label. Typically supplied by tensorflow or keras - unsure of exact format.
            stepcount: number of gradient descent and projection cycles that should be performed.
            stepsize: Callable function that takes in the step number and returns a step size to allow variability. Scales the gradient sign by this amount for each gradient descent step.
            feasibilityProjector: function that takes in an example (np array) and projects it to a case-specific feasibility region. Typically, this is an orthogonal projection onto a linearly defined subspace.
            constrainer: a function which takes in and returns an example as given here, and performs some projection operation to ensure case-specific feasibility. Optional.
            return_labels: return_labels: simple boolean that governs whether or not the labels the model assigns (to the original AND perturbed samples) are returned. optional.

            workercount: How many threads should run in parallel. Recommended to be about half of the running device's thread count. Optional.
            chunksize: chunk size used for the starmap call. Approximately the number of examples assigned to each workrer at a time. Optional.

        Returns:
            Adversaries (large numpy array)
            Optionally, the labels associated with the original and adversarial samples.
    '''

    with multiprocessing.get_context("spawn").Pool(workercount) as p:
        results = p.starmap(constrained_PGD, tqdm.tqdm(zip(
                                repeat(model), dataset, targets, repeat(lossObject), repeat(stepcount), repeat(stepsize), repeat(feasibilityProjector), repeat(constrainer), repeat(return_labels)
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

