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
def EvaluationDispatcher(originalDatasetPath, perturbedDatasetPaths, originalTargetPath, testDataPath, testTargetPath, originalModelPath, retrainedModelPaths, histogramFeatures, descriptors, resultDirectory, computeCorrelation = False)
    '''
        originalDatasetPath: path to original unperturbed dataset. SHould be training data.
        perturbedDatasetPaths: path to perturbed training datasets
        originalTargetPath: training target labels
        testDataPath: testing dataset
        testTargetPath: targets for testing data
        originalModelPath: path to base model
        retrainedModelPaths: list of list of paths to retrained models at different amounts of augmeneted data used for different 
        histogramFeatures: list of feature indices of which histograms should be created 
        descriptors: e.g. ["FGSM", "PGD", "RDSA"]: list of names for each attack

        resultDirectory: Everything will be saved into this folder/appropriate subfolders
        computeCorrelation: allows turning on/off the quite intensive correlation matrix computation.
    '''

    # First, we want to examine how well the attacks did on the base dataset.
    # We will analyze the datasets themselves by computing a few metrics:
        # Cosine similarity between examples and adversaries (create histogram)
        # L1, L2 and L-infinity metric distance between examples and adversaries (create histogram)
        # We will select a random, small handful of features for which we will create histograms across original, FGSM, PGD and RDSA datasets (with different constraints too)
        # Compute, Save and render feature Correlation Matrices (if desired, default is OFF)
    # Render a few histograms


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