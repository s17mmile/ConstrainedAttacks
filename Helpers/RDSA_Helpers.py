import numpy as np

import keras

import tqdm

from keras.losses import mean_squared_error as MSE
from keras.losses import binary_crossentropy as BCE
from keras.losses import categorical_crossentropy as CCE
from scipy import stats

import tensorflow as tf



# Transform a single index (in the range [0, prod(shape)]) into the corresponding multidimensional index.
def transformIndex(index, shape):
    assert index >= 0
    assert index < np.prod(shape)

    transformedIndex = []

    remainder = index

    for dimension in shape:
        coordinate = remainder % dimension
        remainder = int(remainder/dimension)
        transformedIndex.append(coordinate)

    return transformedIndex



def featureContinuity(data, categoricalLimit):
    """
        Counts the number of unique values of each input feature.
        Based on these numbers and a given limit, each variable is considered either continuous or categorical.

        The input need not be two-dimensional - the first dimension will simply be the one to distinguish between examples.
        To index the single features, 
    """

    # To get proper feature counts - since every single number in the array is a feature (for images: each channel for each pixel) - we cannot just use shape[1] for the featureCount and index like that.
    # Instead, we use a small helper (transformIndex) to transform a single index into a multidimensional one for easy looping.
    exampleShape = data.shape[1:]
    print("Detected example shape:", exampleShape)
    featureCount = np.prod(exampleShape)

    allFeatureIndices = [transformIndex(i,exampleShape) for i in range(featureCount)]

    numUniqueValues = [len(np.unique(data[:][index])) for index in allFeatureIndices]

    print(numUniqueValues)

    continuousFeatures = [allFeatureIndices[i] for i, unique_count in enumerate(numUniqueValues) if unique_count > categoricalLimit]
    categoricalFeatures = np.delete(allFeatureIndices, continuousFeatures)

    return numUniqueValues, continuousFeatures, categoricalFeatures

def featureDistributions(data, features, binCount):
    """
        Creates normalized Histograms (= Probability Density Functions) of a (subset of a) dataset's features.
        Practically, these will be the continuous features, as RDSA specifically targets them when shuffling.
        
        Returns Histograms as pairs of two arrays:
            - (binCount+1) bin edges
            - (binCount) bin values
    """

    exampleShape = data.shape[1:]
    featureCount = np.prod(exampleShape)

    print("Initializing histograms.")

    binEdges = np.empty(exampleShape+(binCount+1,))
    probabilities = np.empty(exampleShape+(binCount,))

    print("Calculating Histograms.")

    for featureIndex in tqdm.tqdm(features):
        frequencies, edges = np.histogram(data[:,featureIndex], bins = binCount, density=True)

        binEdges[featureIndex] = edges
        probabilities[featureIndex] = frequencies/sum(frequencies)

    print("Done.")

    return binEdges, probabilities