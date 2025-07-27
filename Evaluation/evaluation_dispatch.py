import numpy as np
import os
import sys

sys.path.append(os.getcwd())

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

import tensorflow as tf
import keras

from dataset_analysis import *
from label_analysis import *

def write_eval_to_file(args)

# This function basically just runs through all the evaluation metrics with a given set of parameters. Then, we juts write separate evaluation configs for each task and attack type and we're good.
# Ideally, this would include all the error handling like the attack dispatcher, btu I'm short on time and cannot be bothered right now.
def EvaluationDispatcher(originalDatasetPath, perturbedDatasetPath, originalTargetPath, testDataPath, testTargetPath, baseModelPath, retrainedModelPaths, histogramFeatures, attackName, resultDirectory, computeCorrelation = False)
    '''
        originalDatasetPath: path to original unperturbed dataset. Should be training data.
        perturbedDatasetPath: path to perturbed training datasets
        originalTargetPath: training target labels
        testDataPath: testing dataset
        testTargetPath: targets for testing data
        baseModelPath: path to base model
        retrainedModelPaths: list of paths to retrained models at different amounts of augmeneted data used
        histogramFeatures: list of feature indices of which histograms should be created 
        attackName: Name for the attack

        resultDirectory: Everything will be saved into this folder/appropriate subfolders
        computeCorrelation: allows turning on/off the quite intensive correlation matrix computation.
    '''

    # Set Up 
    os.makedirs(f"{resultDirectory}/Dataset Metrics")
    os.makedirs(f"{resultDirectory}/Feature Distributions")
    os.makedirs(f"{resultDirectory}/Correlation Plots")
    os.makedirs(f"{resultDirectory}/Retraining Performance")

    # Load data as memmaps
    originalData = np.load(originalDatasetPath, mmap_mode="r")
    perturbedData = np.load(perturbedDatasetPath, mmap_mode="r")

    # Load all the models
    baseModel = keras.models.load_model(baseModelPath)
    retrainedModels = [keras.models.load_model(path) for path in retrainedModelPaths]

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # STEP 1

    # region dataset analysis

    # First, we want to examine how well the attacks did on the base dataset.
    # We will analyze the datasets by computing a few metrics:
        # Cosine similarity between examples and adversaries (create histogram)
        # L1, L2 and L-infinity metric distance between examples and adversaries (create histogram)
        # We will select a random, small handful of features for which we will create histograms across original, FGSM, PGD and RDSA datasets (with different constraints too)
        # Compute, Save and render feature Correlation Matrices (if desired, default is OFF)
    similarity = cosine_similarity(originalData, perturbedData)
    plt.figure(figsize = (16,9))
    plt.hist(similarity, bins = 100, histtype = "step")
    plt.title(f"Cosine Similarity between original and {attackName}-attacked data. Total: ({np.mean(similarity)} ± {np.std(similarity)}")
    plt.savefig(f"{resultDirectory}/Dataset Metrics/{attackName}_cosine_similarity.png")

    l_1 = L_1_norm(perturbedData-originalData)
    plt.figure(figsize = (16,9))
    plt.hist(l_1, bins = 100, histtype = "step")
    plt.title(f"L-1 (manhattan) distance between original data and {attackName}-attacked data. Total: ({np.mean(l_1)} ± {np.std(l_1)}")
    plt.savefig(f"{resultDirectory}/Dataset Metrics/{attackName}_l_1_distance.png")

    l_2 = L_2_norm(perturbedData-originalData)
    plt.figure(figsize = (16,9))
    plt.hist(l_2, bins = 100, histtype = "step")
    plt.title(f"L-2 (euclidean) distance between original data and {attackName}-attacked data. Total: ({np.mean(l_2)} ± {np.std(l_2)}")
    plt.savefig(f"{resultDirectory}/Dataset Metrics/{attackName}_l_2_distance.png")

    l_inf = L_inf_norm(perturbedData-originalData)
    plt.figure(figsize = (16,9))
    plt.hist(l_inf, bins = 100, histtype = "step")
    plt.title(f"L-infinity (max) distance between original data and {attackName}-attacked data. Total: ({np.mean(l_inf)} ± {np.std(l_inf)}")
    plt.savefig(f"{resultDirectory}/Dataset Metrics/{attackName}_l_inf_distance.png")

    # Now, render a few feature histograms
    render_feature_histograms([originalData, perturbedData], ["Original", attackName], histogramFeatures, 100, f"{resultDirectory}/Feature Distributions", attackName)

    # endregion dataset analysis




    # Then, we want to check the performance of the original classifier on the original and perturbed data 
    # We run the classifier on both datasets to obtain original and perturbed TRAINING labels.
    # We can then compute some metrics:
        # Accuracy
        # JSD between original and adversarial training labels
    # We also obtain a "Fooling Matrix"
        # Check for correct classification of example in first and second dataset. Gives 4 options:
            # - Original example incorrect, corresponding adversarial example incorrect ("Robust Negative")
            # - Original example incorrect, corresponding adversarial example correct ("Miracle", should be extremely rare)
            # - Original example correct, corresponding adversarial example incorrect ("Adversary")
            # - Original example correct, corresponding adversarial example correct ("Robust Positive")
    # This matrix gives us the fooling ratio: #Adversaries/(#Adversaries + #Robust Positives)



    # Then, we want to compare the performance of the original and retrained model(s) for each attack type on testing data.
    # We load up the original model and then the retrained models with different amounts of retraining data used.
    
    # We then compute a "Learning Matrix" with these models on the testing dataset:
        # Check for correct classification of example in dataset using both classifiers. Gives 4 options per example:
            # - Original classifier correct, retrained classifier correct ("Consistent Quality")
            # - Original classifier correct, retrained classifier incorrect ("Overcorrect")
            # - Original example incorrect, corresponding adversarial example correct ("Improvement")
            # - Original example incorrect, corresponding adversarial example incorrect ("Consistent Deficit")
    # We can also get the accuracy of these classifiers, which will be our final "well, did it work?" metric.
    # We plot the accuracy and loss vs. the amount of data used for retraining.



    # Write all the important metrics into a file