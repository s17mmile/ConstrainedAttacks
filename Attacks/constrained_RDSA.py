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



def constrained_RDSA(model, example, label, steps, perturbationIndices, binEdges, binProbabilites, constrainer = None):
    '''
        Performs RDSA attack on a single example with a given constraining function (if given).

        Params:
            model: a pre-trained keras model
            example: singular model input. Should be a simple 1D numpy array. Pass by value!
            label: the correct classification label for the given instance --> probability vector (presumably, but not necessarily one-hot!). "Correct" Label is interpreted as the argmax.
            steps: maximum number of shuffling attempts
            perturbationIndices: indices of the variables that should be shuffled.
            binEdges: Array of bin edge vectors (in ascending order) for the variables to be shuffled.
            binProbabilites: Array of probability distributions for the variables to be shuffled.
                --> Each variable will be shuffled by randomly sampling a bin index from the distribution.
                --> Since the perturbed variables are assumed to be continuous, the actual value is chosen uniformly at random from the selected bin.
                --> Note: There is always one more bin edge than there are bins. This will be corrected by randomly choosing a lower bin edge, excluding the highest one.
            constrainer: a function which takes in and returns an example as given here, and performs some projection operation to ensure case-specific feasibility. Optional.

        Returns: Adversary (1D numpy array), the label associated with it and a boolean to indicate fooling success.
            --> If fooling was unsuccessful, return the original example and label 
    '''

    # Initialize a copy of the given example
    adversary = example

    for s in range(steps):
        # Loop over variables to perturb
        for featureIndex in perturbationIndices:

            featureBinEdges = binEdges[featureIndex]

            # Sample probability distribution to get a bin with given prbabilities
            # Choose a lower bin edge at random from all of the bin edges except the highest one.
            low_bin_index = np.random.choice(featureBinEdges.shape[0]-1, p = binProbabilites[featureIndex])

            new_value = np.random.uniform(low = featureBinEdges[low_bin_index], high = featureBinEdges[low_bin_index+1])
            
            adversary[featureIndex] = new_value

        # Apply constraint
        if constrainer is not None:
            adversary = constrainer(adversary)

        # Calculate the model's new prediction. Creates a 1D numpy array containing the probability associated with each class.
        newLabel = model(np.array([adversary]), training = False).numpy()[0]

        # To extract the integer label, we search for the index of the highest entry in the prediction vector.
        if np.argmax(newLabel) != np.argmax(label):
            return adversary, newLabel, True

    # If none of the attempts yielded fooling success, return with a fail state. We might as well still keep track of the adversary as a failed fooling attempt (or rather, one of many). 
    return adversary, newLabel, False

def parallel_constrained_RDSA(model, dataset, labels, steps, perturbationIndexLists, binEdges, binProbabilites, constrainer = None, workercount = 1, chunksize = 4):
    '''
        Performs constrained RDSA attack on a whole set of examples with a given constraining function (if given).
        The use of tqdm means that a progress bar will indicate progress during computation.

        Params:
            model: pre-trained keras model
            dataset: Set of model inputs. 2D numpy array.
            labels: the correct classification label (vector!) for each instance. 2D numpy array.
            steps: maximum number of shuffling attempts. Integer.
            perturbationIndexLists: Lists of feature indices that should be shuffled (different for each example). 2D numpy array.
            binEdges: Array of bin edge vectors (in ascending order) for the variables to be shuffled.
            binProbabilites: Array of probability distributions for the variables to be shuffled.
                --> Each variable will be shuffled by randomly sampling a bin index from the distribution.
                --> Since the perturbed variables are assumed to be continuous, the actual value is chosen uniformly at random from the selected bin.
                --> Note: There is always one more bin edge than there are bins. This will be corrected by randomly choosing a lower bin edge, excluding the highest one.
            constrainer: a function which takes in and returns an example as given here, and performs some projection operation to ensure case-specific feasibility. Optional.

            workercount: How many threads should run in parallel. Recommended to be about half of the running device's thread count. Optional.
            chunksize: chunk size used for the starmap call. Approximately the number of examples assigned to each workrer at a time. Optional.

        Returns three lists:
            Adversaries (numpy arrays)
            The labels associated with them
            Booleans to indicate fooling success.
    '''

    # Return value contains a list for each perturbed example, with:
        # - the perturbed data
        # - the new label given to this data by the model. If this is different from the original, it's a success
        # - a boolean indicating success
    # (The final string goes unused for now, could be used to speed up evaluation)
    adversaries = []
    newLabels = []
    success = []

    with multiprocessing.get_context("spawn").Pool(workercount) as p:
        results = p.starmap(constrained_RDSA, tqdm.tqdm(zip(
                                repeat(model), dataset, labels, repeat(steps), perturbationIndexLists, repeat(binEdges), repeat(binProbabilites), repeat(constrainer)
                            ),
                            total = dataset.shape[0]), chunksize=chunksize)
    
    # Format data for output
    for event in results:
        adversaries.append(event[0])
        newLabels.append(event[1])
        success.append(event[2])

    return np.array(adversaries), np.array(newLabels), np.array(success)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# DEPRECATED

# Runs RDSA attack on a single example with a given constraining projection function (if given).

# 
def old_RDSA(event, model_path, loss_func, steps, var_indices, pdfs_init, bin_idxes_init, bin_edges, unique_values, continuous):
    """
        Method generating adversarial example given a model and a model input

        :param event: A single model input
        :param model_path: File path to the saved deep learning model
        :param loss_func: Loss function used during training the model
        :param steps: Amount of tries to be done for random shuffling
        :param var_indices: Which variables should be perturbed
        :param pdfs_init: Probability vector for the variables to be shuffled
        :param bin_idxes_init: Bin index vector for the variables to be shuffled
        :param bin_edges: Bin edges of the underlying 1D distribution histograms
        :param unique_values: Count of unique values for each variable

        :return: adv: the final adversary after perturbing the input
    """
    
    adv = event[0]
    label = event[1]
    len_var = len(adv)

    pdfs = [[] for i in range(len_var)]
    for i in continuous:
        pdfs[i] = (stats.rv_discrete(values=(bin_idxes_init[i], pdfs_init[i])))
    
    model = keras.models.load_model(model_path)

    for s in range(steps):
        # Go over all specified variables
        for featureIndex in var_indices:
            low_bin = pdfs[featureIndex].rvs(size=1)
            high_bin = low_bin + 1

            if unique_values[featureIndex] < 250:
                new_val = bin_edges[featureIndex][low_bin]
            else:
                new_val = np.random.uniform(low=bin_edges[featureIndex][low_bin],
                                            high=bin_edges[featureIndex][high_bin])[0]
            
            adv[featureIndex] = new_val

        # Stop once classification get fooled
        new_label = np.rint(model(tf.reshape(tf.cast(adv, tf.float32), shape=(1, len_var))))[0]
        if not np.array_equal(new_label, label):
            return adv, new_label, "SUCCESS"

    return adv, new_label, "FAIL"
