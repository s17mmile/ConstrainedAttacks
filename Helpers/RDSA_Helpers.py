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

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DEPRECATED

'''
def GetPDFsAndBinEdges(data, pdfBins):
    """
        Method returning datas underlying (discrete) probability distribution and corresponding bin edges, by creating a
        histogram using a fine binning

        :param data: Underlying data, e.g. entire test data
        :param pdfBins: Amount of bins to be used to discretize probability distributions

        :return: rnd_vars: Probability distributions for each variable
        :return: bin_edges: Bin edges for each variable
    """

    # Get array of all variables
    variables = np.asarray([data[:, i] for i in range(len(data[0]))])

    # For each variable get "pdf" to be able to correctly sample new values
    rnd_vars = []
    bin_edges = []
    pdfs = []
    bin_idxes = []
    for i in range(len(variables)):
        if len(np.unique(variables[i])) < 50:
            # dens, bins = np.histogram(variables[i], bins=len(np.unique(variables[i]))-1, density=True)
            bins, dens = np.unique(variables[i], return_counts=True)
            dens = np.divide(dens, len(variables[i]))
        elif len(np.unique(variables[i])) < 100:
            dens, bins = np.histogram(variables[i], bins=50 - 1, density=True)
        else:
            dens, bins = np.histogram(variables[i], bins=pdfBins - 1, density=True)

        pdf = dens / sum(dens)

        if len(np.unique(variables[i])) < 50:
            bin_idx = np.arange(0, len(np.unique(variables[i])))
        elif len(np.unique(variables[i])) < pdfBins:
            bin_idx = np.arange(0, 50 - 1)
        else:
            bin_idx = np.arange(0, pdfBins - 1)

        bin_edges.append(bins)
        pdfs.append(pdf)
        bin_idxes.append(bin_idx)
        #rnd_vars.append(stats.rv_discrete(values=(bin_idx, pdf)))

    return pdfs, bin_idxes, bin_edges

def DistributionShuffleAttackPerVarsParallel(event, model_path, loss_func, steps, var_indices, pdfs_init, bin_idxes_init,
                                             bin_edges, unique_values, continuous):
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
    
    numFeatures = len(event[0])

    pdfs = [[] for i in range(numFeatures)]
    for i in continuous:
        pdfs[i] = (stats.rv_discrete(values=(bin_idxes_init[i], pdfs_init[i])))

    input = event[0]
    label = event[1]
    len_var = len(input)
    input = tf.reshape(tf.cast(input, tf.float32), shape=(1, len_var))
    
    model = keras.models.load_model(model_path)
    adv = input

    for s in range(steps):
        adv = tf.reshape(tf.cast(adv, tf.float32), shape=(1, len_var))
        pred = model(adv)

        adv = adv.numpy()[0]

        if loss_func == "BCE":
            label = tf.reshape(label, (1, pred.shape[-1]))
            loss = BCE(label, pred)
        elif loss_func == "CCE":
            label = tf.reshape(label, (1, pred.shape[-1]))
            loss = CCE(label, pred)
        else:
            loss = MSE(label, pred)

        # Go over all specified variables
        for i in range(len(var_indices)):
            low_bin = pdfs[var_indices[i]].rvs(size=1)
            high_bin = low_bin + 1

            if unique_values[var_indices[i]] < 250:
                new_val = bin_edges[var_indices[i]][low_bin]

            else:
                new_val = np.random.uniform(low=bin_edges[var_indices[i]][low_bin],
                                            high=bin_edges[var_indices[i]][high_bin])[0]
            adv[var_indices[i]] = new_val

        # Stop once classification get fooled
        if not np.array_equal(
                np.rint(model(tf.reshape(tf.cast(adv, tf.float32), shape=(1, len_var))))[0],
                label.numpy()[0]):
            return adv

    return adv
'''