import numpy as np
import os
import sys
import tqdm

# Catch unnecessary sklearn warnings. Not super clean.
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

sys.path.append(os.getcwd())

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

import tensorflow as tf
import keras



from Evaluation.evaluation_dispatch import EvaluationDispatcher 

# Hard-Coded number of retraining subdivions
retraining_subdivisions = 3

# Selector panel
# CIFAR, ImageNet and MNIST will run (unless otherwise specified) using the scaled data - no further constraints.
CIFAR_FGSM_scaled = True
CIFAR_PGD_scaled = True
CIFAR_RDSA_scaled = True

CIFAR_FGSM_scaled_boxed = True
CIFAR_PGD_scaled_boxed = True
CIFAR_RDSA_scaled_boxed = False



ImageNet_FGSM_scaled = True
ImageNet_PGD_scaled = True
ImageNet_RDSA_scaled = True

ImageNet_FGSM_scaled_boxed = True
ImageNet_PGD_scaled_boxed = True
ImageNet_RDSA_scaled_boxed = True



MNIST_FGSM_scaled = True
MNIST_PGD_scaled = True
MNIST_RDSA_scaled = True

MNIST_FGSM_scaled_boxed = True
MNIST_PGD_scaled_boxed = True
MNIST_RDSA_scaled_boxed = True



# TopoDNN has more options. These can be seen in run_attacks.py with a bit more explanation.
TopoDNN_FGSM_clip = True
TopoDNN_PGD_clip = True
TopoDNN_RDSA_clip = True

TopoDNN_FGSM_constits_clip = True
TopoDNN_RDSA_constits_clip = True
TopoDNN_PGD_constits_clip = True

TopoDNN_FGSM_constits_clip_globalEnergy = True
TopoDNN_PGD_constits_clip_globalEnergy = True
TopoDNN_RDSA_constits_clip_globalEnergy = True

TopoDNN_FGSM_constits_clip_particleEnergy = True
TopoDNN_PGD_constits_clip_particleEnergy = True
TopoDNN_RDSA_constits_clip_particleEnergy = True

if __name__ == "__main__":

    # Run eval with all the different configs

    # With retraining
    # region CIFAR_scaled

    if (CIFAR_FGSM_scaled):
        try:
            baseModelPath="Models/CIFAR10/base_model.keras"
            attackName = "FGSM_scaled"
            retrainedModelPaths = [os.path.join(os.path.dirname(baseModelPath), attackName, os.path.basename(baseModelPath).replace(".keras", f"_retrained_{i}.keras")) for i in range(retraining_subdivisions)]

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/CIFAR10/train_data.npy",
                perturbedDatasetPath="Adversaries/CIFAR10/scaled/FGSM_train_data.npy",
                originalTargetPath="Datasets/CIFAR10/train_target.npy",
                testDataPath="Datasets/CIFAR10/test_data.npy",
                testTargetPath="Datasets/CIFAR10/test_target.npy",
                baseModelPath="Models/CIFAR10/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [(15,15,0),(15,15,1),(15,15,2)],
                attackName = attackName,
                resultDirectory=f"Results/CIFAR10/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (CIFAR_PGD_scaled):
        try:
            baseModelPath="Models/CIFAR10/base_model.keras"
            attackName = "PGD_scaled"
            retrainedModelPaths = [os.path.join(os.path.dirname(baseModelPath), attackName, os.path.basename(baseModelPath).replace(".keras", f"_retrained_{i}.keras")) for i in range(retraining_subdivisions)]

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/CIFAR10/train_data.npy",
                perturbedDatasetPath="Adversaries/CIFAR10/scaled/PGD_train_data.npy",
                originalTargetPath="Datasets/CIFAR10/train_target.npy",
                testDataPath="Datasets/CIFAR10/test_data.npy",
                testTargetPath="Datasets/CIFAR10/test_target.npy",
                baseModelPath="Models/CIFAR10/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [(15,15,0),(15,15,1),(15,15,2)],
                attackName = attackName,
                resultDirectory=f"Results/CIFAR10/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (CIFAR_RDSA_scaled):
        try:
            baseModelPath="Models/CIFAR10/base_model.keras"
            attackName = "RDSA_scaled"
            retrainedModelPaths = [os.path.join(os.path.dirname(baseModelPath), attackName, os.path.basename(baseModelPath).replace(".keras", f"_retrained_{i}.keras")) for i in range(retraining_subdivisions)]

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/CIFAR10/train_data.npy",
                perturbedDatasetPath="Adversaries/CIFAR10/scaled/RDSA_train_data.npy",
                originalTargetPath="Datasets/CIFAR10/train_target.npy",
                testDataPath="Datasets/CIFAR10/test_data.npy",
                testTargetPath="Datasets/CIFAR10/test_target.npy",
                baseModelPath="Models/CIFAR10/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [(15,15,0),(15,15,1),(15,15,2)],
                attackName = attackName,
                resultDirectory=f"Results/CIFAR10/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    # endregion CIFAR_scaled

    # No retraining here
    # region CIFAR_scaled_boxed

    if (CIFAR_FGSM_scaled_boxed):
        try:
            baseModelPath="Models/CIFAR10/base_model.keras"
            attackName = "FGSM_scaled_boxed"
            retrainedModelPaths = []

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/CIFAR10/train_data.npy",
                perturbedDatasetPath="Adversaries/CIFAR10/scaled_boxed/FGSM_train_data.npy",
                originalTargetPath="Datasets/CIFAR10/train_target.npy",
                testDataPath="Datasets/CIFAR10/test_data.npy",
                testTargetPath="Datasets/CIFAR10/test_target.npy",
                baseModelPath="Models/CIFAR10/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [(15,15,0),(15,15,1),(15,15,2)],
                attackName = attackName,
                resultDirectory=f"Results/CIFAR10/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (CIFAR_PGD_scaled_boxed):
        try:
            baseModelPath="Models/CIFAR10/base_model.keras"
            attackName = "PGD_scaled_boxed"
            retrainedModelPaths = []

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/CIFAR10/train_data.npy",
                perturbedDatasetPath="Adversaries/CIFAR10/scaled_boxed/PGD_train_data.npy",
                originalTargetPath="Datasets/CIFAR10/train_target.npy",
                testDataPath="Datasets/CIFAR10/test_data.npy",
                testTargetPath="Datasets/CIFAR10/test_target.npy",
                baseModelPath="Models/CIFAR10/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [(15,15,0),(15,15,1),(15,15,2)],
                attackName = attackName,
                resultDirectory=f"Results/CIFAR10/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (CIFAR_RDSA_scaled_boxed):
        try:
            baseModelPath="Models/CIFAR10/base_model.keras"
            attackName = "RDSA_scaled_boxed"
            retrainedModelPaths = []

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/CIFAR10/train_data.npy",
                perturbedDatasetPath="Adversaries/CIFAR10/scaled_boxed/RDSA_train_data.npy",
                originalTargetPath="Datasets/CIFAR10/train_target.npy",
                testDataPath="Datasets/CIFAR10/test_data.npy",
                testTargetPath="Datasets/CIFAR10/test_target.npy",
                baseModelPath="Models/CIFAR10/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [(15,15,0),(15,15,1),(15,15,2)],
                attackName = attackName,
                resultDirectory=f"Results/CIFAR10/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    # endregion CIFAR_scaled_boxed



    # No retraining here
    # region ImageNet_scaled

    if (ImageNet_FGSM_scaled):
        try:
            baseModelPath="Models/ImageNet/base_model.keras"
            attackName = "FGSM_scaled"
            retrainedModelPaths = []

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/ImageNet/train_data.npy",
                perturbedDatasetPath="Adversaries/ImageNet/scaled/FGSM_train_data.npy",
                originalTargetPath="Datasets/ImageNet/train_target.npy",
                testDataPath="Datasets/ImageNet/test_data.npy",
                testTargetPath="Datasets/ImageNet/test_target.npy",
                baseModelPath="Models/ImageNet/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/ImageNet/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (ImageNet_PGD_scaled):
        try:
            baseModelPath="Models/ImageNet/base_model.keras"
            attackName = "PGD_scaled"
            retrainedModelPaths = []

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/ImageNet/train_data.npy",
                perturbedDatasetPath="Adversaries/ImageNet/scaled/PGD_train_data.npy",
                originalTargetPath="Datasets/ImageNet/train_target.npy",
                testDataPath="Datasets/ImageNet/test_data.npy",
                testTargetPath="Datasets/ImageNet/test_target.npy",
                baseModelPath="Models/ImageNet/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/ImageNet/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (ImageNet_RDSA_scaled):
        try:
            baseModelPath="Models/ImageNet/base_model.keras"
            attackName = "RDSA_scaled"
            retrainedModelPaths = []

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/ImageNet/train_data.npy",
                perturbedDatasetPath="Adversaries/ImageNet/scaled/RDSA_train_data.npy",
                originalTargetPath="Datasets/ImageNet/train_target.npy",
                testDataPath="Datasets/ImageNet/test_data.npy",
                testTargetPath="Datasets/ImageNet/test_target.npy",
                baseModelPath="Models/ImageNet/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/ImageNet/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    # endregion ImageNet_scaled

    # No retraining here
    # region ImageNet_scaled_boxed

    if (ImageNet_FGSM_scaled_boxed):
        try:
            baseModelPath="Models/ImageNet/base_model.keras"
            attackName = "FGSM_scaled_boxed"
            retrainedModelPaths = []

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/ImageNet/train_data.npy",
                perturbedDatasetPath="Adversaries/ImageNet/scaled_boxed/FGSM_train_data.npy",
                originalTargetPath="Datasets/ImageNet/train_target.npy",
                testDataPath="Datasets/ImageNet/test_data.npy",
                testTargetPath="Datasets/ImageNet/test_target.npy",
                baseModelPath="Models/ImageNet/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/ImageNet/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (ImageNet_PGD_scaled_boxed):
        try:
            baseModelPath="Models/ImageNet/base_model.keras"
            attackName = "PGD_scaled_boxed"
            retrainedModelPaths = []

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/ImageNet/train_data.npy",
                perturbedDatasetPath="Adversaries/ImageNet/scaled_boxed/PGD_train_data.npy",
                originalTargetPath="Datasets/ImageNet/train_target.npy",
                testDataPath="Datasets/ImageNet/test_data.npy",
                testTargetPath="Datasets/ImageNet/test_target.npy",
                baseModelPath="Models/ImageNet/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/ImageNet/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (ImageNet_RDSA_scaled_boxed):
        try:
            baseModelPath="Models/ImageNet/base_model.keras"
            attackName = "RDSA_scaled_boxed"
            retrainedModelPaths = []

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/ImageNet/train_data.npy",
                perturbedDatasetPath="Adversaries/ImageNet/scaled_boxed/RDSA_train_data.npy",
                originalTargetPath="Datasets/ImageNet/train_target.npy",
                testDataPath="Datasets/ImageNet/test_data.npy",
                testTargetPath="Datasets/ImageNet/test_target.npy",
                baseModelPath="Models/ImageNet/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/ImageNet/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    # endregion ImageNet_scaled_boxed



    # region MNIST_scaled

    if (MNIST_FGSM_scaled):
        try:
            baseModelPath="Models/MNIST/base_model.keras"
            attackName = "FGSM_scaled"
            retrainedModelPaths = [os.path.join(os.path.dirname(baseModelPath), attackName, os.path.basename(baseModelPath).replace(".keras", f"_retrained_{i}.keras")) for i in range(retraining_subdivisions)]

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/MNIST/train_data.npy",
                perturbedDatasetPath="Adversaries/MNIST/scaled/FGSM_train_data.npy",
                originalTargetPath="Datasets/MNIST/train_target.npy",
                testDataPath="Datasets/MNIST/test_data.npy",
                testTargetPath="Datasets/MNIST/test_target.npy",
                baseModelPath="Models/MNIST/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/MNIST/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (MNIST_PGD_scaled):
        try:
            baseModelPath="Models/MNIST/base_model.keras"
            attackName = "PGD_scaled"
            retrainedModelPaths = [os.path.join(os.path.dirname(baseModelPath), attackName, os.path.basename(baseModelPath).replace(".keras", f"_retrained_{i}.keras")) for i in range(retraining_subdivisions)]

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/MNIST/train_data.npy",
                perturbedDatasetPath="Adversaries/MNIST/scaled/PGD_train_data.npy",
                originalTargetPath="Datasets/MNIST/train_target.npy",
                testDataPath="Datasets/MNIST/test_data.npy",
                testTargetPath="Datasets/MNIST/test_target.npy",
                baseModelPath="Models/MNIST/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/MNIST/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (MNIST_RDSA_scaled):
        try:
            baseModelPath="Models/MNIST/base_model.keras"
            attackName = "RDSA_scaled"
            retrainedModelPaths = [os.path.join(os.path.dirname(baseModelPath), attackName, os.path.basename(baseModelPath).replace(".keras", f"_retrained_{i}.keras")) for i in range(retraining_subdivisions)]

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/MNIST/train_data.npy",
                perturbedDatasetPath="Adversaries/MNIST/scaled/RDSA_train_data.npy",
                originalTargetPath="Datasets/MNIST/train_target.npy",
                testDataPath="Datasets/MNIST/test_data.npy",
                testTargetPath="Datasets/MNIST/test_target.npy",
                baseModelPath="Models/MNIST/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/MNIST/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    # endregion MNIST_scaled
    
    # No retraining here
    # region MNIST_scaled_boxed

    if (MNIST_FGSM_scaled_boxed):
        try:
            baseModelPath="Models/MNIST/base_model.keras"
            attackName = "FGSM_scaled_boxed"
            retrainedModelPaths = []

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/MNIST/train_data.npy",
                perturbedDatasetPath="Adversaries/MNIST/scaled_boxed/FGSM_train_data.npy",
                originalTargetPath="Datasets/MNIST/train_target.npy",
                testDataPath="Datasets/MNIST/test_data.npy",
                testTargetPath="Datasets/MNIST/test_target.npy",
                baseModelPath="Models/MNIST/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/MNIST/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (MNIST_PGD_scaled_boxed):
        try:
            baseModelPath="Models/MNIST/base_model.keras"
            attackName = "PGD_scaled_boxed"
            retrainedModelPaths = []

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/MNIST/train_data.npy",
                perturbedDatasetPath="Adversaries/MNIST/scaled_boxed/PGD_train_data.npy",
                originalTargetPath="Datasets/MNIST/train_target.npy",
                testDataPath="Datasets/MNIST/test_data.npy",
                testTargetPath="Datasets/MNIST/test_target.npy",
                baseModelPath="Models/MNIST/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/MNIST/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (MNIST_RDSA_scaled_boxed):
        try:
            baseModelPath="Models/MNIST/base_model.keras"
            attackName = "RDSA_scaled_boxed"
            retrainedModelPaths = []

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/MNIST/train_data.npy",
                perturbedDatasetPath="Adversaries/MNIST/scaled_boxed/RDSA_train_data.npy",
                originalTargetPath="Datasets/MNIST/train_target.npy",
                testDataPath="Datasets/MNIST/test_data.npy",
                testTargetPath="Datasets/MNIST/test_target.npy",
                baseModelPath="Models/MNIST/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/MNIST/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    # endregion MNIST_scaled_boxed



    # region TopoDNN_clip

    if (TopoDNN_FGSM_clip):
        try:
            baseModelPath="Models/TopoDNN/base_model.keras"
            attackName = "FGSM_spreadLimit"
            retrainedModelPaths = [os.path.join(os.path.dirname(baseModelPath), attackName, os.path.basename(baseModelPath).replace(".keras", f"_retrained_{i}.keras")) for i in range(retraining_subdivisions)]

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/TopoDNN/train_data.npy",
                perturbedDatasetPath="Adversaries/TopoDNN/spreadLimit/FGSM_train_data.npy",
                originalTargetPath="Datasets/TopoDNN/train_target.npy",
                testDataPath="Datasets/TopoDNN/test_data.npy",
                testTargetPath="Datasets/TopoDNN/test_target.npy",
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/TopoDNN/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (TopoDNN_PGD_clip):
        try:
            baseModelPath="Models/TopoDNN/base_model.keras"
            attackName = "PGD_spreadLimit"
            retrainedModelPaths = [os.path.join(os.path.dirname(baseModelPath), attackName, os.path.basename(baseModelPath).replace(".keras", f"_retrained_{i}.keras")) for i in range(retraining_subdivisions)]

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/TopoDNN/train_data.npy",
                perturbedDatasetPath="Adversaries/TopoDNN/spreadLimit/PGD_train_data.npy",
                originalTargetPath="Datasets/TopoDNN/train_target.npy",
                testDataPath="Datasets/TopoDNN/test_data.npy",
                testTargetPath="Datasets/TopoDNN/test_target.npy",
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/TopoDNN/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (TopoDNN_RDSA_clip):
        try:
            baseModelPath="Models/TopoDNN/base_model.keras"
            attackName = "RDSA_spreadLimit"
            retrainedModelPaths = [os.path.join(os.path.dirname(baseModelPath), attackName, os.path.basename(baseModelPath).replace(".keras", f"_retrained_{i}.keras")) for i in range(retraining_subdivisions)]

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/TopoDNN/train_data.npy",
                perturbedDatasetPath="Adversaries/TopoDNN/spreadLimit/RDSA_train_data.npy",
                originalTargetPath="Datasets/TopoDNN/train_target.npy",
                testDataPath="Datasets/TopoDNN/test_data.npy",
                testTargetPath="Datasets/TopoDNN/test_target.npy",
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/TopoDNN/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    # endregion TopoDNN_clip

    

    # region TopoDNN_constits_clip

    if (TopoDNN_FGSM_constits_clip):
        try:
            baseModelPath="Models/TopoDNN/base_model.keras"
            attackName = "FGSM_conserveConstits_spreadLimit"
            retrainedModelPaths = [os.path.join(os.path.dirname(baseModelPath), attackName, os.path.basename(baseModelPath).replace(".keras", f"_retrained_{i}.keras")) for i in range(retraining_subdivisions)]

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/TopoDNN/train_data.npy",
                perturbedDatasetPath="Adversaries/TopoDNN/conserveConstits_spreadLimit/FGSM_train_data.npy",
                originalTargetPath="Datasets/TopoDNN/train_target.npy",
                testDataPath="Datasets/TopoDNN/test_data.npy",
                testTargetPath="Datasets/TopoDNN/test_target.npy",
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/TopoDNN/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (TopoDNN_PGD_constits_clip):
        try:
            baseModelPath="Models/TopoDNN/base_model.keras"
            attackName = "PGD_conserveConstits_spreadLimit"
            retrainedModelPaths = [os.path.join(os.path.dirname(baseModelPath), attackName, os.path.basename(baseModelPath).replace(".keras", f"_retrained_{i}.keras")) for i in range(retraining_subdivisions)]

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/TopoDNN/train_data.npy",
                perturbedDatasetPath="Adversaries/TopoDNN/conserveConstits_spreadLimit/PGD_train_data.npy",
                originalTargetPath="Datasets/TopoDNN/train_target.npy",
                testDataPath="Datasets/TopoDNN/test_data.npy",
                testTargetPath="Datasets/TopoDNN/test_target.npy",
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/TopoDNN/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (TopoDNN_RDSA_constits_clip):
        try:
            baseModelPath="Models/TopoDNN/base_model.keras"
            attackName = "RDSA_conserveConstits_spreadLimit"
            retrainedModelPaths = [os.path.join(os.path.dirname(baseModelPath), attackName, os.path.basename(baseModelPath).replace(".keras", f"_retrained_{i}.keras")) for i in range(retraining_subdivisions)]

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/TopoDNN/train_data.npy",
                perturbedDatasetPath="Adversaries/TopoDNN/conserveConstits_spreadLimit/RDSA_train_data.npy",
                originalTargetPath="Datasets/TopoDNN/train_target.npy",
                testDataPath="Datasets/TopoDNN/test_data.npy",
                testTargetPath="Datasets/TopoDNN/test_target.npy",
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/TopoDNN/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    # endregion TopoDNN_constits_clip



    # region TopoDNN_constits_clip_globalEnergy

    if (TopoDNN_FGSM_constits_clip_globalEnergy):
        try:
            baseModelPath="Models/TopoDNN/base_model.keras"
            attackName = "FGSM_conserveConstits_spreadLimit_conserveGlobalEnergy"
            retrainedModelPaths = [os.path.join(os.path.dirname(baseModelPath), attackName, os.path.basename(baseModelPath).replace(".keras", f"_retrained_{i}.keras")) for i in range(retraining_subdivisions)]

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/TopoDNN/train_data.npy",
                perturbedDatasetPath="Adversaries/TopoDNN/conserveConstits_spreadLimit_conserveGlobalEnergy/FGSM_train_data.npy",
                originalTargetPath="Datasets/TopoDNN/train_target.npy",
                testDataPath="Datasets/TopoDNN/test_data.npy",
                testTargetPath="Datasets/TopoDNN/test_target.npy",
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/TopoDNN/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (TopoDNN_PGD_constits_clip_globalEnergy):
        try:
            baseModelPath="Models/TopoDNN/base_model.keras"
            attackName = "PGD_conserveConstits_spreadLimit_conserveGlobalEnergy"
            retrainedModelPaths = [os.path.join(os.path.dirname(baseModelPath), attackName, os.path.basename(baseModelPath).replace(".keras", f"_retrained_{i}.keras")) for i in range(retraining_subdivisions)]

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/TopoDNN/train_data.npy",
                perturbedDatasetPath="Adversaries/TopoDNN/conserveConstits_spreadLimit_conserveGlobalEnergy/PGD_train_data.npy",
                originalTargetPath="Datasets/TopoDNN/train_target.npy",
                testDataPath="Datasets/TopoDNN/test_data.npy",
                testTargetPath="Datasets/TopoDNN/test_target.npy",
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/TopoDNN/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (TopoDNN_RDSA_constits_clip_globalEnergy):
        try:
            baseModelPath="Models/TopoDNN/base_model.keras"
            attackName = "RDSA_conserveConstits_spreadLimit_conserveGlobalEnergy"
            retrainedModelPaths = [os.path.join(os.path.dirname(baseModelPath), attackName, os.path.basename(baseModelPath).replace(".keras", f"_retrained_{i}.keras")) for i in range(retraining_subdivisions)]

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/TopoDNN/train_data.npy",
                perturbedDatasetPath="Adversaries/TopoDNN/conserveConstits_spreadLimit_conserveGlobalEnergy/RDSA_train_data.npy",
                originalTargetPath="Datasets/TopoDNN/train_target.npy",
                testDataPath="Datasets/TopoDNN/test_data.npy",
                testTargetPath="Datasets/TopoDNN/test_target.npy",
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/TopoDNN/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    # endregion TopoDNN_constits_clip_globalEnergy


    
    # region TopoDNN_constits_clip_particleEnergy

    if (TopoDNN_FGSM_constits_clip_particleEnergy):
        try:
            baseModelPath="Models/TopoDNN/base_model.keras"
            attackName = "FGSM_conserveConstits_spreadLimit_conserveParticleEnergy"
            retrainedModelPaths = [os.path.join(os.path.dirname(baseModelPath), attackName, os.path.basename(baseModelPath).replace(".keras", f"_retrained_{i}.keras")) for i in range(retraining_subdivisions)]

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/TopoDNN/train_data.npy",
                perturbedDatasetPath="Adversaries/TopoDNN/conserveConstits_spreadLimit_conserveParticleEnergy/FGSM_train_data.npy",
                originalTargetPath="Datasets/TopoDNN/train_target.npy",
                testDataPath="Datasets/TopoDNN/test_data.npy",
                testTargetPath="Datasets/TopoDNN/test_target.npy",
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/TopoDNN/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (TopoDNN_PGD_constits_clip_particleEnergy):
        try:
            baseModelPath="Models/TopoDNN/base_model.keras"
            attackName = "PGD_conserveConstits_spreadLimit_conserveParticleEnergy"
            retrainedModelPaths = [os.path.join(os.path.dirname(baseModelPath), attackName, os.path.basename(baseModelPath).replace(".keras", f"_retrained_{i}.keras")) for i in range(retraining_subdivisions)]

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/TopoDNN/train_data.npy",
                perturbedDatasetPath="Adversaries/TopoDNN/conserveConstits_spreadLimit_conserveParticleEnergy/PGD_train_data.npy",
                originalTargetPath="Datasets/TopoDNN/train_target.npy",
                testDataPath="Datasets/TopoDNN/test_data.npy",
                testTargetPath="Datasets/TopoDNN/test_target.npy",
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/TopoDNN/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (TopoDNN_RDSA_constits_clip_particleEnergy):
        try:
            baseModelPath="Models/TopoDNN/base_model.keras"
            attackName = "RDSA_conserveConstits_spreadLimit_conserveParticleEnergy"
            retrainedModelPaths = [os.path.join(os.path.dirname(baseModelPath), attackName, os.path.basename(baseModelPath).replace(".keras", f"_retrained_{i}.keras")) for i in range(retraining_subdivisions)]

            print(retrainedModelPaths)

            EvaluationDispatcher(
                originalDatasetPath="Datasets/TopoDNN/train_data.npy",
                perturbedDatasetPath="Adversaries/TopoDNN/conserveConstits_spreadLimit_conserveParticleEnergy/RDSA_train_data.npy",
                originalTargetPath="Datasets/TopoDNN/train_target.npy",
                testDataPath="Datasets/TopoDNN/test_data.npy",
                testTargetPath="Datasets/TopoDNN/test_target.npy",
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainedModelPaths=retrainedModelPaths,
                histogramFeatures = [],
                attackName = attackName,
                resultDirectory=f"Results/TopoDNN/{attackName}",
                computeCorrelation=True
            )
        except Exception as e:
            print(f"Failure: {e}")

    # endregion TopoDNN_constits_clip_particleEnergy