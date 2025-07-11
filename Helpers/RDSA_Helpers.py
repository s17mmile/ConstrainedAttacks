import numpy as np

import keras

from keras.losses import mean_squared_error as MSE
from keras.losses import binary_crossentropy as BCE
from keras.losses import categorical_crossentropy as CCE
from scipy import stats

import tensorflow as tf

def featureContinuity(data, categoricalLimit):
    """
        Counts the number of unique values of each input feature.
        Based on these numbers and a given limit, each variable is considered either continuous or categorical.

        The input data should be of the shape [#EXAMPLES] x [#FEATURES].
            --> data[n,:] gives the array representing example n.
            --> data[:,f] gives the array representing (the distribution of) feature f.
            --> data[n,f] gives the value of feature f in example n.
    """

    featureCount = data.shape[1]

    numUniqueValues = [len(np.unique(data[:, i])) for i in range(featureCount)]
    continuousFeatures = [i for i, unique_count in enumerate(numUniqueValues) if unique_count > categoricalLimit]
    categoricalFeatures = np.delete(np.arange(0, featureCount), continuousFeatures)

    return numUniqueValues, continuousFeatures, categoricalFeatures

def featureDistributions(dataset, features, binCount):
    """
        Creates normalized Histograms (= Probability Density Functions) of a (subset of a) dataset's features.
        Practically, these will be the continuous features, as RDSA specifically targets them when shuffling.

        The input dataset should be of shape [#EXAMPLES] x [#FEATURES].
            --> dataset[n,:] gives the array representing example n.
            --> dataset[:,f] gives the array representing (the distribution of) feature f.
        
        Returns Histograms as pairs of two arrays:
            - (binCount+1) bin edges
            - (binCount) bin values
    """

    numFeatures = dataset.shape[1]
    binEdges = [[] for i in range(numFeatures)]
    probabilities = [[] for i in range(numFeatures)]

    for featureIndex in features:
        frequencies, edges = np.histogram(dataset[:,featureIndex], bins = binCount, density=True)

        binEdges[featureIndex] = edges
        probabilities[featureIndex] = frequencies/sum(frequencies)

    return binEdges, probabilities