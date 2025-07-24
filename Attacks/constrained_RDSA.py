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

def constrained_RDSA(model, example, target, steps, perturbationIndices, binEdges, binProbabilites, constrainer = None, return_labels = False):
    '''
        Performs RDSA attack on a single example with a given constraining function (if given).

        Params:
            model: a pre-trained keras model
            example: singular model input as a numpy array. Can be multidimensional.
            target: the correct classification label for the given instance (one-hot)
            steps: maximum number of shuffling attempts
            perturbationIndices: indices of the variables that should be shuffled.
            binEdges: Array of bin edge vectors (in ascending order) for the variables to be shuffled.
            binProbabilites: Array of probability distributions for the variables to be shuffled.
                --> Each variable will be shuffled by randomly sampling a bin index from the distribution.
                --> Since the perturbed variables are assumed to be continuous, the actual value is chosen uniformly at random from the selected bin.
                --> Note: There is always one more bin edge than there are bins. This will be corrected by randomly choosing a lower bin edge, excluding the highest one.
            constrainer: a function which takes in and returns an example as given here, and performs some projection operation to ensure case-specific feasibility. Optional.
            return_labels: return_labels: simple boolean that governs whether or not the labels the model assigns (to the original AND perturbed samples) are returned.

        Returns: Adversary and, optionally, the label associated with the original and adversarial sample.
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


        # Calculate the model's new prediction. Creates a 1D numpy array containing the probability associated with each class.
        newLabel = model(np.array([adversary]), training = False).numpy()[0]

        # To extract the integer label, we search for the index of the highest entry in the prediction vector.
        if np.argmax(newLabel) != np.argmax(target):
            break



    # If given: apply the final constrainer.
    # This can result in a decrease of the loss function, there's not really any way to avoid that.
    # We essentially hope that adding the constraint doesn't fix the prediction.
    if constrainer is not None:
        adversary = constrainer(adversary)

    # Compte and return the labels if wanted
    if (return_labels):
        originalLabel = model(tf.convert_to_tensor([example]), training = False).numpy()[0]
        newLabel = model(tf.convert_to_tensor([adversary]), training = False).numpy()[0]
        return adversary, originalLabel, newLabel
    else:
        return adversary



def parallel_constrained_RDSA(model, dataset, targets, steps, categoricalFeatureMaximum, binCount, perturbedFeatureCount, constrainer = None, return_labels = False, workercount = 1, chunksize = 4, n = None):
    '''
        Performs constrained RDSA attack on a whole set of examples with a given constraining function (if given).
        The use of tqdm means that a progress bar will indicate progress during computation.

        Params:
            model: pre-trained keras model
            dataset: Set of model inputs. 2D numpy array.
            targets: the correct classification label (vector!) for each instance. 2D numpy array.
            steps: maximum number of shuffling attempts. Integer.
            constrainer: a function which takes in and returns an example as given here, and performs some projection operation to ensure case-specific feasibility. Optional.
            return_labels: return_labels: simple boolean that governs whether or not the labels the model assigns (to the original AND perturbed samples) are returned. Optional.
            
            workercount: How many threads should run in parallel. Recommended to be about half of the running device's thread count. Optional.
            chunksize: chunk size used for the starmap call. Approximately the number of examples assigned to each workrer at a time. Optional.
            n: If given, only the first n examples will be perturbed. If None, all examples will be perturbed. Optional.

        Returns two lists:
            Adversaries (numpy arrays)
            The labels associated with them
    '''

    '''
        STEP 1: RDSA Preparation

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



    # Limit to first n examples if n is given
    dataset = dataset[:n]
    targets = targets[:n]

    with multiprocessing.get_context("spawn").Pool(workercount) as p:
        results = p.starmap(constrained_RDSA, tqdm.tqdm(zip(
                                repeat(model), dataset, targets, repeat(steps), perturbationIndexLists, repeat(binEdges), repeat(binProbabilites), repeat(constrainer), repeat(return_labels)
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