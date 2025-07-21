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

import Helpers.RDSA_Helpers as RDSA_Help

def constrained_RDSA(model, example, label, steps, perturbationIndices, binEdges, binProbabilites, constrainer = None):
    '''
        Performs RDSA attack on a single example with a given constraining function (if given).

        Params:
            model: a pre-trained keras model
            example: singular model input as a numpy array. Can be multidimensional.
            label: the correct classification label for the given instance --> probability vector. "Correct" Label is interpreted as the argmax.
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

            featureBinEdges = binEdges[*featureIndex]
            featureBinProbabilities = binProbabilites[*featureIndex]

            # Sample probability distribution to get a bin with given prbabilities
            # Choose a lower bin edge at random from all of the bin edges except the highest one.
            low_bin_index = np.random.choice(featureBinEdges.shape[0]-1, p = featureBinProbabilities.flatten())

            new_value = np.random.uniform(low = featureBinEdges[low_bin_index], high = featureBinEdges[low_bin_index+1])
            
            adversary[*featureIndex] = new_value

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

def parallel_constrained_RDSA(model, dataset, labels, steps, categoricalFeatureMaximum, binCount, perturbedFeatureCount, constrainer = None, workercount = 1, chunksize = 4, n = None):
    '''
        Performs constrained RDSA attack on a whole set of examples with a given constraining function (if given).
        The use of tqdm means that a progress bar will indicate progress during computation.

        Params:
            model: pre-trained keras model
            dataset: Set of model inputs. 2D numpy array.
            labels: the correct classification label (vector!) for each instance. 2D numpy array.
            steps: maximum number of shuffling attempts. Integer.

            constrainer: a function which takes in and returns an example as given here, and performs some projection operation to ensure case-specific feasibility. Optional.

            workercount: How many threads should run in parallel. Recommended to be about half of the running device's thread count. Optional.
            chunksize: chunk size used for the starmap call. Approximately the number of examples assigned to each workrer at a time. Optional.
            n: If given, only the first n examples will be perturbed. If None, all examples will be perturbed. Optional.

        Returns three lists:
            Adversaries (numpy arrays)
            The labels associated with them
            Booleans to indicate fooling success.
    '''

    # STEP 1: RDSA Preparation

    '''
        Constructs the following:
            perturbationIndexLists: Lists of feature indices that should be shuffled (different for each example). 2D numpy array.
            binEdges: Array of bin edge vectors (in ascending order) for the variables to be shuffled.
            binProbabilites: Array of probability distributions for the variables to be shuffled.
                --> Each variable will be shuffled by randomly sampling a bin index from the distribution.
                --> Since the perturbed variables are assumed to be continuous, the actual value is chosen uniformly at random from the selected bin.
                --> Note: There is always one more bin edge than there are bins. This will be corrected by randomly choosing a lower bin edge, excluding the highest one.
    '''

    # n is the number of examples to perturb. If not given, set it to the number of examples in the dataset.
    if n is None:
        n = dataset.shape[0]

    # Find indices of features to be considered continuous/categorical.
    numUniqueValues, continuous, categorical = RDSA_Help.featureContinuity(dataset, categoricalFeatureMaximum)

    # Generate probability density function for each continuous feature.
    # Non-continuous features are given an empty placeholder
    binEdges, binProbabilites  = RDSA_Help.featureDistributions(dataset, continuous, binCount)

    # Randomly choose a given number of continuous features to be perturbed for the first n examples
    perturbationIndexLists = [random.sample(continuous, perturbedFeatureCount) for i in range(n)]



    # Step 2: Return value contains a list for each perturbed example, with:
        # - the perturbed data
        # - the new label given to this data by the model. If this is different from the original, it's a success
        # - a boolean indicating success

    adversaries = []
    newLabels = []
    success = []

    # Limit to first n examples if n is given
    dataset = dataset[:n]
    labels = labels[:n]

    with multiprocessing.get_context("spawn").Pool(workercount) as p:
        results = p.starmap(constrained_RDSA, tqdm.tqdm(zip(
                                repeat(model), dataset, labels, repeat(steps), perturbationIndexLists, repeat(binEdges), repeat(binProbabilites), repeat(constrainer)
                            ),
                            total = dataset.shape[0]), chunksize=chunksize)
    
    # Format data for output
    print("Formatting results...")
    for event in results:
        adversaries.append(event[0])
        newLabels.append(event[1])
        success.append(event[2])

    return np.array(adversaries), np.array(newLabels), np.array(success)