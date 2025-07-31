import numpy as np
import os
import sys

sys.path.append(os.getcwd())

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

import tensorflow as tf
import keras

from Evaluation.dataset_analysis import *
from Evaluation.label_analysis import *

# Small printing nicety
def center_text(x, symbolcount):
    string = str(x)

    if len(string) >= symbolcount: return string

    if (symbolcount-len(string)) % 2 == 1:
        string = string+" "

    padding = int((symbolcount-len(string))/2)

    return padding*" "+string+padding*" "

# This function basically just runs through all the evaluation metrics with a given set of parameters. Then, we juts write separate evaluation configs for each task and attack type and we're good.
# Ideally, this would include all the error handling like the attack dispatcher, btu I'm short on time and cannot be bothered right now.
def EvaluationDispatcher(originalDatasetPath, perturbedDatasetPath, originalTargetPath, testDataPath, testTargetPath, baseModelPath, retrainedModelPaths, histogramFeatures, attackName, resultDirectory, computeCorrelation = False):
    '''
        originalDatasetPath: path to original unperturbed dataset. Should be training data.
        perturbedDatasetPath: path to perturbed training datasets
        originalTargetPath: training target labels
        testDataPath: testing dataset
        testTargetPath: targets for testing data
        baseModelPath: path to base model
        retrainedModelPaths: list (!) of paths to retrained models at different amounts of augmeneted data used. Just pass [] if no retraining analysis should be performed.
        histogramFeatures: list of feature indices of which histograms should be created 
        attackName: Name for the attack

        resultDirectory: Everything will be saved into this folder/appropriate subfolders
        computeCorrelation: allows turning on/off the quite intensive correlation matrix computation.
    '''

    # region setup

    print(f"Analysis of attack {attackName} on dataset {originalDatasetPath}.")

    print("SETUP")

    # Set Up 
    try:
        os.makedirs(resultDirectory)
    except:
        pass

    try:
        os.makedirs(f"{resultDirectory}/Dataset Metrics")
    except:
        pass

    try:
        os.makedirs(f"{resultDirectory}/Feature Distributions")
    except:
        pass

    try:
        os.makedirs(f"{resultDirectory}/Model Performance")
    except:
        pass

    if computeCorrelation:
        try:
            os.makedirs(f"{resultDirectory}/Correlation Plots")
        except:
            pass

    # Load data as memmaps
    print("LOAD")

    originalData = np.load(originalDatasetPath, mmap_mode="r")
    perturbedData = np.load(perturbedDatasetPath, mmap_mode="r")
    originalTarget = np.load(originalTargetPath, mmap_mode="r")

    testData = np.load(testDataPath, mmap_mode="r")
    testTarget = np.load(testTargetPath, mmap_mode="r")

    # Load all the models
    baseModel = keras.models.load_model(baseModelPath)
    retrainedModels = [keras.models.load_model(path) for path in retrainedModelPaths]

    # endregion setup

    

    # region dataset analysis

    print("DATASET")

    # First, we want to examine how well the attacks did on the base dataset.
    # We will analyze the datasets by computing a few metrics:
        # Cosine similarity between examples and adversaries (create histogram)
        # L1, L2 and L-infinity metric distance between examples and adversaries (create histogram)
        # We will select a random, small handful of features for which we will create histograms across original, FGSM, PGD and RDSA datasets (with different constraints too)
        # Compute, Save and render feature Correlation Matrices (if desired, default is OFF)
    similarity = cosine_similarity(originalData, perturbedData)
    plt.figure(figsize = (16,9))
    plt.hist(similarity, bins = 100, histtype = "step")
    plt.title(f"Cosine Similarity between original and {attackName}-attacked data. Total: ({np.mean(similarity)} ± {np.std(similarity)})")
    plt.grid()
    plt.savefig(f"{resultDirectory}/Dataset Metrics/{attackName}_cosine_similarity.png")
    plt.close()

    l_1 = L_1_norm(perturbedData-originalData)
    plt.figure(figsize = (16,9))
    plt.hist(l_1, bins = 100, histtype = "step")
    plt.title(f"L-1 (manhattan) distance between original data and {attackName}-attacked data. Total: ({np.mean(l_1)} ± {np.std(l_1)})")
    plt.grid()
    plt.savefig(f"{resultDirectory}/Dataset Metrics/{attackName}_l_1_distance.png")
    plt.close()

    l_2 = L_2_norm(perturbedData-originalData)
    plt.figure(figsize = (16,9))
    plt.hist(l_2, bins = 100, histtype = "step")
    plt.title(f"L-2 (euclidean) distance between original data and {attackName}-attacked data. Total: ({np.mean(l_2)} ± {np.std(l_2)})")
    plt.grid()
    plt.savefig(f"{resultDirectory}/Dataset Metrics/{attackName}_l_2_distance.png")
    plt.close()

    l_inf = L_inf_norm(perturbedData-originalData)
    plt.figure(figsize = (16,9))
    plt.hist(l_inf, bins = 100, histtype = "step")
    plt.title(f"L-infinity (max) distance between original data and {attackName}-attacked data. Total: ({np.mean(l_inf)} ± {np.std(l_inf)})")
    plt.grid()
    plt.savefig(f"{resultDirectory}/Dataset Metrics/{attackName}_l_inf_distance.png")
    plt.close()

    # Compute dataset JSD (for this one, it doesn't make much sense to go feature by feature and create Histograms, I think. Too hard to glean info from them.)
    print("JSD")
    adversarial_data_JSD = dataset_JSD(perturbedData, originalData)

    # Now, render a few feature histograms. Which exactly is given by the histogramFeatures list/array.
    print("HISTOGRAMS")
    render_feature_histograms([originalData, perturbedData], ["Original", attackName], histogramFeatures, 100, f"{resultDirectory}/Feature Distributions", attackName)

    # Finally, add correlation plot if desired. Not recommended for particularly large input sizes.
    print("CORRELATION")
    if computeCorrelation:
        render_correlation_matrix(originalData,f"{resultDirectory}/Correlation Plots/original_correlation.png", "original")
        render_correlation_matrix(perturbedData,f"{resultDirectory}/Correlation Plots/{attackName}_correlation.png", attackName)

    # endregion dataset analysis



    # region fooling

    # Then, we want to check the performance of the original classifier on the original and perturbed data 
    # We run the classifier on both datasets to obtain original and perturbed TRAINING labels.
    print("PREDICTION")
    original_base_labels = baseModel.predict(originalData)
    perturbed_base_labels = baseModel.predict(perturbedData)
    
    # We can then compute some metrics
    # Accuracy
    print("ACCURACY")
    original_base_accuracy = accuracy(original_base_labels, originalTarget)
    perturbed_base_accuracy = accuracy(perturbed_base_labels, originalTarget)

    # Get and plot per-class accuracy: which classes are easier/harder to properly classify?
    print("ACCURACY PER CLASS")
    original_base_accuracy_per_class = accuracy_per_class(original_base_labels, originalTarget)
    perturbed_base_accuracy_per_class = accuracy_per_class(perturbed_base_labels, originalTarget)

    num_classes = original_base_labels.shape[1]
    step_edges = np.arange(-0.5, num_classes+0.5)

    plt.figure(figsize = (16,9))
    plt.stairs(original_base_accuracy_per_class, step_edges, label = "Base Model accuracy per class on original data.")
    plt.stairs(perturbed_base_accuracy_per_class, step_edges, label = "Base Model accuracy per class on perturbed data.")
    plt.title(f"{attackName}: Accuracy per class using original classifier for original and perturbed dataset.")
    plt.xlim(-0.5, num_classes-0.5)
    plt.ylim(0,1)
    plt.legend()
    plt.grid()
    plt.savefig(f"{resultDirectory}/Dataset Metrics/{attackName}_accuracy_per_class.png")
    plt.close()

    # JSD between original labels, adversarial labels and target labels
    print("LABEL JSD")
    base_original_target_JSD = JSD(original_base_labels, originalTarget)
    base_perturbed_target_JSD = JSD(perturbed_base_labels, originalTarget)
    base_original_perturbed_JSD = JSD(original_base_labels, perturbed_base_labels)

    # Obtain and render confusion matrices comparing the original labels and perturbed labels with the target.
    print("CONFUSION")
    target_original_confusion_matrix = label_confusion_matrix(originalTarget, original_base_labels)
    plt.figure(figsize = (16,9))
    plt.imshow(target_original_confusion_matrix, cmap="coolwarm", aspect = 1)
    plt.tick_params(bottom = False, top = True)
    plt.colorbar()
    plt.title(f"Confusion Matrix: Classifier labels for unperturbed training data VS. target labels.")
    plt.xlabel(f"Unperturbed Label")
    plt.ylabel(f"Target Label")
    plt.savefig(f"{resultDirectory}/Dataset Metrics/{attackName}_original_target_confusion.png")
    plt.close()

    target_perturbed_confusion_matrix = label_confusion_matrix(originalTarget, perturbed_base_labels)
    plt.figure(figsize = (16,9))
    plt.imshow(target_perturbed_confusion_matrix, cmap="coolwarm", aspect = 1)
    plt.tick_params(bottom = False, top = True)
    plt.colorbar()
    plt.title(f"Confusion Matrix: Classifier labels for perturbed training data VS. target labels.")
    plt.xlabel(f"Perturbed Label")
    plt.ylabel(f"Target Label")
    plt.savefig(f"{resultDirectory}/Dataset Metrics/{attackName}_perturbed_target_confusion.png")
    plt.close()

    original_perturbed_comparison_matrix = label_confusion_matrix(original_base_labels, perturbed_base_labels)
    plt.figure(figsize = (16,9))
    plt.imshow(original_perturbed_comparison_matrix, cmap="coolwarm", aspect = 1)
    plt.tick_params(bottom = False, top = True)
    plt.colorbar()
    plt.title(f"Confusion Matrix: Classifier labels for perturbed VS. unperturbed training data.")
    plt.xlabel(f"Perturbed Label")
    plt.ylabel(f"Original Label")
    plt.savefig(f"{resultDirectory}/Dataset Metrics/{attackName}_original_perturbed_confusion.png")
    plt.close()
    


    # We also obtain a "Fooling Matrix", essentially a confusion matrix of correctness.
        # Check for correct classification of example in first and second dataset. Gives 4 options:
            # - Index [0,0]: Original example incorrect, corresponding adversarial example incorrect ("Robust Negative")
            # - Index [0,1]: Original example incorrect, corresponding adversarial example correct ("Miracle", should be extremely rare)
            # - Index [1,0]: Original example correct, corresponding adversarial example incorrect ("Adversary")
            # - Index [1,1]: Original example correct, corresponding adversarial example correct ("Robust Positive")

    print("FOOLING")
    fooling_matrix = get_fooling_matrix(original_base_labels, perturbed_base_labels, originalTarget)

    # This matrix gives us the fooling ratio: #Adversaries/(#Adversaries + #Robust Positives)
    fooling_ratio = fooling_matrix[1,0]/(fooling_matrix[1,0] + fooling_matrix[1,1])

    # Might as well calculate the ration of misclassified events that were fixed by the attack. Intuitively, this should be zero.
    miracle_ratio = fooling_matrix[0,1]/(fooling_matrix[0,0] + fooling_matrix[0,1])

    # endregion fooling



    # region retrained model eval

    if len(retrainedModelPaths) != 0:
        print("RETRAINED")
        # Then, we want to compare the performance of the original and retrained model(s) for each attack type on testing data.
        # We use the original model and then the retrained models with different amounts of retraining data used.

        print("PREDICTIONS")
        test_base_labels = baseModel.predict(testData)
        test_retrained_labels = [model.predict(testData) for model in retrainedModels]

        # We then compute a "Learning Matrix" (principally identical to the fooling matrix) with these models on the testing dataset:
            # Check for correct classification of example in dataset using both classifiers. Gives 4 options per example:
                # - Index [0,0]: Original classifier incorrect, retrained classifier incorrect ("Consistent Deficit")
                # - Index [0,1]: Original classifier incorrect, retrained classifier correct ("Improvement")
                # - Index [1,0]: Original classifier correct, retrained classifier incorrect ("Overcorrect")
                # - Index [1,1]: Original classifier correct, retrained classifier correct ("Consistent Quality")
        print("LEARNING")
        learning_matrices = [get_learning_matrix(test_base_labels, labels, testTarget) for labels in test_retrained_labels]

        # Improvement ratio: what fraction of the previously misclassified examples are now correct?
        improvement_ratios = [matrix[0,1]/(matrix[0,1] + matrix[0,0]) for matrix in learning_matrices]

        # Overcorrection ratio: what fraction of the previously correctly classified examples are now incorrect?
        overcorrect_ratios = [matrix[1,0]/(matrix[1,0] + matrix[1,1]) for matrix in learning_matrices]

        # We can also plot the accuracy and loss of these classifiers.
        # This assumes that the retrained models were trained using evenly-increasing amounts of adversarial data. This is given in this project.
        # So, we just use the percentage of data used in retraining as the x axis.
        print("ACC_LOSS_PLOTTING")
        x_vals = np.linspace(0, 1, len(retrainedModels)+1, endpoint=True)

        # Accuracy (will be our final "well, did it work?" metric)
        test_base_accuracy = accuracy(test_base_labels, testTarget)
        test_retrained_accuracies = [accuracy(labels, testTarget) for labels in test_retrained_labels]

        accuracy_vals = [test_base_accuracy]+test_retrained_accuracies

        plt.figure(figsize=(16,9))
        plt.xlim(0,1)
        plt.xlabel("Fraction of adversarial training data used for training")
        plt.ylim(0,1)
        plt.ylabel("Accuracy on Testing dataset")
        plt.title("Dependence of Accuracy on amount of adversarial retraining Data")
        plt.scatter(x_vals, accuracy_vals, label = "Accuracy vs. Amount of retraining Data used")
        plt.legend()
        plt.grid()
        plt.savefig(f"{resultDirectory}/Model Performance/{attackName}_Retraining_Accuracy.png")
        plt.close()

        # Loss
        test_base_loss = baseModel.loss(testTarget, test_base_labels)
        test_retrained_losses = [model.loss(testTarget, labels) for model, labels in zip(retrainedModels, test_retrained_labels)]

        loss_vals = [test_base_loss]+test_retrained_losses

        plt.figure(figsize=(16,9))
        plt.xlim(0,1)
        plt.xlabel("Fraction of adversarial training data used for training")
        # plt.ylim(0,1)
        plt.ylabel("Loss on Testing dataset")
        plt.title("Dependence of Training Loss on amount of adversarial retraining Data")
        plt.scatter(x_vals, loss_vals, label = "Loss vs. Amount of retraining Data used")
        plt.legend()
        plt.grid()
        plt.savefig(f"{resultDirectory}/Model Performance/{attackName}_Retraining_Loss.png")    
        plt.close()

        # Compute and save AUROC diagrams and values
        print("AUROC")
        base_auroc_score = renderROCandGetAUROC(test_base_labels, testTarget, f"{resultDirectory}/Model Performance/{attackName}_ROC_curve_base.png", attackName)

        for i, labels in enumerate(test_retrained_labels):
            retrained_auroc_scores = [renderROCandGetAUROC(labels, testTarget, f"{resultDirectory}/Model Performance/{attackName}_ROC_curve_{i}.png", attackName) for i in range(len(retrainedModels))]

    # endregion retrained model eval

    print("WRITING FILE")

    # Write all the important metric averages that can't be nicely visualized into a file
    with open(f"{resultDirectory}/{attackName}_evaluation.txt", "w") as f:
        f.write(f"Evaluation of {attackName} attack\n")
        f.write(f"\n")
        f.write(f"\n")
        f.write(f"\n")
        f.write(f"Adversarial Dataset Evaluation:\n")
        f.write(f"Average Cosine Similarity between original and perturbed Data Samples: ({np.mean(similarity)} ± {np.std(similarity)}).\n")
        f.write(f"Average L1 Distance between original and perturbed Data Samples: ({np.mean(l_1)} ± {np.std(l_1)}).\n")
        f.write(f"Average L2 Distance between original and perturbed Data Samples: ({np.mean(l_2)} ± {np.std(l_2)}).\n")
        f.write(f"Average L-infinity Distance between original and perturbed Data Samples: ({np.mean(l_inf)} ± {np.std(l_inf)}).\n")
        f.write(f"Overall Jensen-Shannon Distance between original and perturbed Datasets: {adversarial_data_JSD}.\n")
        f.write(f"\n")
        f.write(f"\n")
        f.write(f"\n")
        f.write(f"Model Fooling Metrics:\n")
        f.write(f"Accuracy of Original Model on unmodified training data: {original_base_accuracy}.\n")
        f.write(f"Accuracy of Original Model on perturbed training data: {perturbed_base_accuracy}.\n")
        f.write(f"\n")
        f.write(f"Jensen Shannon Distances\n")
        f.write(f" - between original dataset labels and target labels:            {base_original_target_JSD}\n")
        f.write(f" - between perturbed dataset labels and target labels:           {base_perturbed_target_JSD}\n")
        f.write(f" - between original dataset labels and perturbed dataset labels: {base_original_perturbed_JSD}\n")
        f.write(f"\n")
        f.write(f"Fooling Matrix:\n")
        f.write(f"                              | Perturbed Label is incorrect |  Perturbed Label is correct  \n")
        f.write(f"------------------------------|------------------------------|------------------------------\n")
        f.write(f" Original Label is incorrect  |{center_text(fooling_matrix[0,0], 30)}|{center_text(fooling_matrix[0,1], 30)}\n")
        f.write(f"------------------------------|------------------------------|------------------------------\n")
        f.write(f" Original Label is correct    |{center_text(fooling_matrix[1,0], 30)}|{center_text(fooling_matrix[1,1], 30)}\n")
        f.write(f"\n")
        f.write(f"Fooling Ratio: {fooling_ratio}.\n")
        f.write(f"Miracle Ratio: {miracle_ratio}.\n")
        f.write(f"\n")
        f.write(f"\n")
        f.write(f"\n")

        # File output omitted if retraining analysis is skipped.
        if len(retrainedModelPaths) != 0:
            f.write(f"------------------------------------------------------------------------------------------------------------\n")
            f.write(f"\n")
            f.write(f"Retraining Metrics:\n")
            f.write(f"\n")
            f.write(f"Base Test Accuracy: {test_base_accuracy}\n")
            f.write(f"Base Test Loss: {test_base_loss}\n")
            f.write(f"Base AUROC: {base_auroc_score}\n")
            for i in range(len(retrainedModels)):
                matrix = learning_matrices[i]
                f.write(f"\n")
                f.write(f"\n")
                f.write(f"\n")
                f.write(f"Learning Matrix of Retrained Model with {100*(i+1)/len(retrainedModels)}% adversarial Data\n")
                f.write(f"                              | Retrained Model is incorrect |  Retrained Model is correct  \n")
                f.write(f"------------------------------|------------------------------|------------------------------\n")
                f.write(f" Original Model is incorrect  |{center_text(matrix[0,0], 30)}|{center_text(matrix[0,1], 30)}\n")
                f.write(f"------------------------------|------------------------------|------------------------------\n")
                f.write(f" Original Model is correct    |{center_text(matrix[1,0], 30)}|{center_text(matrix[1,1], 30)}\n")
                f.write(f"\n")
                f.write(f"Improvement Ratio: {improvement_ratios[i]}\n")
                f.write(f"Overcorrect Ratio: {overcorrect_ratios[i]}\n")
                f.write(f"\n")
                f.write(f"Accuracy: {test_retrained_accuracies[i]}\n")
                f.write(f"Loss: {test_retrained_losses[i]}\n")
                f.write(f"AUROC Score: {retrained_auroc_scores[i]}\n")

    

    print("\n\n")