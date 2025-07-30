import numpy as np
import os
import sys
import warnings
import scipy.spatial
import tqdm
import matplotlib.pyplot as plt
import scipy

sys.path.append(os.getcwd())

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

import tensorflow as tf
import keras

from Helpers.RDSA_Helpers import featureDistributions

# Element-wise (<=> one for each example) cosine similarity between original and modified dataset.
def cosine_similarity(original_data, adversarial_data):
    return np.array([np.dot(example.flatten(), adversary.flatten())/(np.linalg.norm(example.flatten()) * np.linalg.norm(adversary.flatten())) for example, adversary in zip(original_data, adversarial_data)])

# Element-wise L-1-Norm for a set of examples
# L-1-norm is the same as the sum of absolutes
def L_1_norm(data):
    return np.sum(np.abs(data), axis = 1)

# Element-wise L-2-Norm for a set of examples
# L-2-Norm is just Euclidean Distance
def L_2_norm(data):
    return np.sqrt(np.sum(np.square(data), axis = 1))

# Element-wise L-infinity-Norm for a set of examples
# L-2-Norm is just the maximum magnitude of any value
def L_inf_norm(data):
    return np.max(np.abs(data), axis = 1)

# Jensen Shannon Distance between the feature Distributions
def dataset_JSD(dataset1, dataset2):

    # Get single feature distributions for both datasets. Will sort into 100 bins, since this is the default value in featureDistributions().
    # We could also just get the frequencies (without the normalization applied in featureDistributions()), but this is easier.
    # The shape of these arrays will then be {INPUT SHAPE}x100
    _, probabilities1 = featureDistributions(dataset1)
    _, probabilities2 = featureDistributions(dataset1)

    print(probabilities1.shape)
    print(probabilities2.shape)

    # When computing JSD, flatten the frequency arrays, putting all the histograms in one long line.
    probabilities1 = probabilities1.flatten()
    probabilities2 = probabilities2.flatten()

    return scipy.spatial.distance.jensenshannon(probabilities1, probabilities2)

# Create and render histograms of a given subset of a dataset's features, saving to a given directory.
# This actually takes in multiple datasets, and will create the histograms for each in the same plot with different colors.
def render_feature_histograms(datasets, datasetNames, features, binCount, output_directory, out_name):

    for featureIndex in features:
        plt.figure(figsize = (16,9))

        # Load feature data from memory maps (will also work with non-memmapped data, but might be inefficient)
        # Can't unwrap an int, so we need to make a small exception for 1D inputs and not attempt an unwrap. Makes list comprehension ugly so I split it.
        feature_data = []
        for dataset in datasets:
            if dataset.ndim == 2:
                feature_data.append(dataset[:,featureIndex])
            else:
                feature_data.append(dataset[:,*featureIndex])

        plt.hist(feature_data, bins = binCount, histtype = "step", label = datasetNames)

        plt.legend()
        plt.savefig(f"{output_directory}/{out_name}_{featureIndex}.png")

        plt.close()
    
    return


# Create and render Pearson product-moment correlation coefficients between all variables in a dataset.
# Since this can be computationally expensive (quadratic in number of input features!), we give the option to save the correlations array for later use.
def render_correlation_matrix(dataset, output_path, array_path = None):
    # Each example in the dataset needs to be "flattened", meaning the dataset is squished into two dimensions
    flat_data = np.reshape(dataset, (dataset.shape[0], int(dataset.size/dataset.shape[0])))

    # We use rowvar = False because each feature is given in a column - each row represents a sample.
    correlations = np.corrcoef(flat_data, rowvar = False)

    plt.figure(figsize = (16,9))
    plt.tick_params(bottom = False, top = True)
    plt.imshow(correlations, vmin=-1, vmax=1, cmap="coolwarm", interpolation = None)

    plt.savefig(output_path, dpi = 300)

    if array_path is not None:
        np.save(array_path, correlations)

    plt.close()

    return