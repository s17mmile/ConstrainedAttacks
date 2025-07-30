import numpy as np
import os
import sys

sys.path.append(os.getcwd())

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

import tensorflow as tf
import keras

from Helpers.retraining_dispatch import RetrainingDispatcher

# Selector Panel: Choose which retrainings to perform
CIFAR_FGSM_scaled = False
CIFAR_PGD_scaled = False
CIFAR_RDSA_scaled = False

CIFAR_FGSM_scaled_boxed = False
CIFAR_PGD_scaled_boxed = False
CIFAR_RDSA_scaled_boxed = False



MNIST_FGSM_scaled = False
MNIST_PGD_scaled = False
MNIST_RDSA_scaled = False

MNIST_FGSM_scaled_boxed = False
MNIST_PGD_scaled_boxed = False
MNIST_RDSA_scaled_boxed = False



ImageNet_FGSM_scaled = False
ImageNet_PGD_scaled = False
ImageNet_RDSA_scaled = False

ImageNet_FGSM_scaled_boxed = False
ImageNet_PGD_scaled_boxed = False
ImageNet_RDSA_scaled_boxed = False



TopoDNN_FGSM_clip = True
TopoDNN_PGD_clip = True
TopoDNN_RDSA_clip = True

TopoDNN_FGSM_constits_clip = True
TopoDNN_PGD_constits_clip = True
TopoDNN_RDSA_constits_clip = True

TopoDNN_FGSM_constits_clip_globalEnergy = True
TopoDNN_PGD_constits_clip_globalEnergy = True
TopoDNN_RDSA_constits_clip_globalEnergy = True

TopoDNN_FGSM_constits_clip_particleEnergy = True
TopoDNN_PGD_constits_clip_particleEnergy = True
TopoDNN_RDSA_constits_clip_particleEnergy = True



# I just globally set the number of retraining subdivisions to 5 - this should be enough to see some trends.
retraining_subdivisions = 5

# Tgis should be enough to trigger the early stop at some point (patience 3). If not, ah well.
epochs = 20

if __name__ == "__main__":

    # region cifar

    if(CIFAR_FGSM_scaled):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/CIFAR10/base_model.keras",
                retrainingDataPath="Adversaries/CIFAR10/scaled/FGSM_train_data.npy",
                trainingTargetPath="Datasets/CIFAR10/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="FGSM_scaled"
            )
        except Exception as e:
            print(f"Failure: {e}")

    if(CIFAR_PGD_scaled):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/CIFAR10/base_model.keras",
                retrainingDataPath="Adversaries/CIFAR10/scaled/PGD_train_data.npy",
                trainingTargetPath="Datasets/CIFAR10/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="PGD_scaled"
            )
        except Exception as e:
            print(f"Failure: {e}")

    if(CIFAR_RDSA_scaled):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/CIFAR10/base_model.keras",
                retrainingDataPath="Adversaries/CIFAR10/scaled/RDSA_train_data.npy",
                trainingTargetPath="Datasets/CIFAR10/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="RDSA_scaled"
            )
        except Exception as e:
            print(f"Failure: {e}")



    if(CIFAR_FGSM_scaled_boxed):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/CIFAR10/base_model.keras",
                retrainingDataPath="Adversaries/CIFAR10/scaled_boxed/FGSM_train_data.npy",
                trainingTargetPath="Datasets/CIFAR10/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="FGSM_scaled_boxed"
            )
        except Exception as e:
            print(f"Failure: {e}")

    if(CIFAR_PGD_scaled_boxed):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/CIFAR10/base_model.keras",
                retrainingDataPath="Adversaries/CIFAR10/scaled_boxed/PGD_train_data.npy",
                trainingTargetPath="Datasets/CIFAR10/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="PGD_scaled_boxed"
            )
        except Exception as e:
            print(f"Failure: {e}")

    if(CIFAR_RDSA_scaled_boxed):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/CIFAR10/base_model.keras",
                retrainingDataPath="Adversaries/CIFAR10/scaled_boxed/RDSA_train_data.npy",
                trainingTargetPath="Datasets/CIFAR10/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="RDSA_scaled_boxed"
            )
        except Exception as e:
            print(f"Failure: {e}")

    # endregion cifar



    # region MNIST

    if(MNIST_FGSM_scaled):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/MNIST/maxpool_model.keras",
                retrainingDataPath="Adversaries/MNIST/scaled/FGSM_train_data.npy",
                trainingTargetPath="Datasets/MNIST/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="FGSM_scaled"
            )
        except Exception as e:
            print(f"Failure: {e}")

    if(MNIST_PGD_scaled):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/MNIST/maxpool_model.keras",
                retrainingDataPath="Adversaries/MNIST/scaled/PGD_train_data.npy",
                trainingTargetPath="Datasets/MNIST/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="PGD_scaled"
            )
        except Exception as e:
            print(f"Failure: {e}")

    if(MNIST_RDSA_scaled):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/MNIST/maxpool_model.keras",
                retrainingDataPath="Adversaries/MNIST/scaled/RDSA_train_data.npy",
                trainingTargetPath="Datasets/MNIST/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="RDSA_scaled"
            )
        except Exception as e:
            print(f"Failure: {e}")



    if(MNIST_FGSM_scaled_boxed):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/MNIST/maxpool_model.keras",
                retrainingDataPath="Adversaries/MNIST/scaled_boxed/FGSM_train_data.npy",
                trainingTargetPath="Datasets/MNIST/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="FGSM_scaled_boxed"
            )
        except Exception as e:
            print(f"Failure: {e}")

    if(MNIST_PGD_scaled_boxed):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/MNIST/maxpool_model.keras",
                retrainingDataPath="Adversaries/MNIST/scaled_boxed/PGD_train_data.npy",
                trainingTargetPath="Datasets/MNIST/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="PGD_scaled_boxed"
            )
        except Exception as e:
            print(f"Failure: {e}")

    if(MNIST_RDSA_scaled_boxed):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/MNIST/maxpool_model.keras",
                retrainingDataPath="Adversaries/MNIST/scaled_boxed/RDSA_train_data.npy",
                trainingTargetPath="Datasets/MNIST/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="RDSA_scaled_boxed"
            )
        except Exception as e:
            print(f"Failure: {e}")

    # endregion MNIST



    # region ImageNet

    if(ImageNet_FGSM_scaled):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/ImageNet/base_model.keras",
                retrainingDataPath="Adversaries/ImageNet/scaled/FGSM_threshold_data.npy",
                trainingTargetPath="Datasets/ImageNet/threshold_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="FGSM_scaled"
            )
        except Exception as e:
            print(f"Failure: {e}")

    if(ImageNet_PGD_scaled):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/ImageNet/base_model.keras",
                retrainingDataPath="Adversaries/ImageNet/scaled/PGD_threshold_data.npy",
                trainingTargetPath="Datasets/ImageNet/threshold_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="PGD_scaled"
            )
        except Exception as e:
            print(f"Failure: {e}")

    if(ImageNet_RDSA_scaled):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/ImageNet/base_model.keras",
                retrainingDataPath="Adversaries/ImageNet/scaled/RDSA_threshold_data.npy",
                trainingTargetPath="Datasets/ImageNet/threshold_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="RDSA_scaled"
            )
        except Exception as e:
            print(f"Failure: {e}")



    if(ImageNet_FGSM_scaled_boxed):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/ImageNet/base_model.keras",
                retrainingDataPath="Adversaries/ImageNet/scaled_boxed/FGSM_threshold_data.npy",
                trainingTargetPath="Datasets/ImageNet/threshold_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="FGSM_scaled_boxed"
            )
        except Exception as e:
            print(f"Failure: {e}")

    if(ImageNet_PGD_scaled_boxed):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/ImageNet/base_model.keras",
                retrainingDataPath="Adversaries/ImageNet/scaled_boxed/PGD_threshold_data.npy",
                trainingTargetPath="Datasets/ImageNet/threshold_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="PGD_scaled_boxed"
            )
        except Exception as e:
            print(f"Failure: {e}")

    if(ImageNet_RDSA_scaled_boxed):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/ImageNet/base_model.keras",
                retrainingDataPath="Adversaries/ImageNet/scaled_boxed/RDSA_threshold_data.npy",
                trainingTargetPath="Datasets/ImageNet/threshold_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="RDSA_scaled_boxed"
            )
        except Exception as e:
            print(f"Failure: {e}")

    # endregion ImageNet



    # region TopoDNN

    if(TopoDNN_FGSM_clip):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainingDataPath="Adversaries/TopoDNN/spreadLimit/FGSM_train_data.npy",
                trainingTargetPath="Datasets/TopoDNN/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="FGSM_spreadLimit"
            )
        except Exception as e:
            print(f"Failure: {e}")

    if(TopoDNN_PGD_clip):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainingDataPath="Adversaries/TopoDNN/spreadLimit/PGD_train_data.npy",
                trainingTargetPath="Datasets/TopoDNN/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="PGD_spreadLimit"
            )
        except Exception as e:
            print(f"Failure: {e}")

    if(TopoDNN_RDSA_clip):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainingDataPath="Adversaries/TopoDNN/spreadLimit/RDSA_train_data.npy",
                trainingTargetPath="Datasets/TopoDNN/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="RDSA_spreadLimit"
            )
        except Exception as e:
            print(f"Failure: {e}")



    if(TopoDNN_FGSM_constits_clip):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainingDataPath="Adversaries/TopoDNN/conserveConstits_spreadLimit/FGSM_train_data.npy",
                trainingTargetPath="Datasets/TopoDNN/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="FGSM_conserveConstits_spreadLimit"
            )
        except Exception as e:
            print(f"Failure: {e}")

    if(TopoDNN_PGD_constits_clip):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainingDataPath="Adversaries/TopoDNN/conserveConstits_spreadLimit/PGD_train_data.npy",
                trainingTargetPath="Datasets/TopoDNN/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="PGD_conserveConstits_spreadLimit"
            )
        except Exception as e:
            print(f"Failure: {e}")

    if(TopoDNN_RDSA_constits_clip):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainingDataPath="Adversaries/TopoDNN/conserveConstits_spreadLimit/RDSA_train_data.npy",
                trainingTargetPath="Datasets/TopoDNN/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="RDSA_conserveConstits_spreadLimit"
            )
        except Exception as e:
            print(f"Failure: {e}")



    if(TopoDNN_FGSM_constits_clip_globalEnergy):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainingDataPath="Adversaries/TopoDNN/conserveConstits_spreadLimit_conserveGlobalEnergy/FGSM_train_data.npy",
                trainingTargetPath="Datasets/TopoDNN/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="FGSM_conserveConstits_spreadLimit_conserveGlobalEnergy"
            )
        except Exception as e:
            print(f"Failure: {e}")

    if(TopoDNN_PGD_constits_clip_globalEnergy):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainingDataPath="Adversaries/TopoDNN/conserveConstits_spreadLimit_conserveGlobalEnergy/PGD_train_data.npy",
                trainingTargetPath="Datasets/TopoDNN/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="PGD_conserveConstits_spreadLimit_conserveGlobalEnergy"
            )
        except Exception as e:
            print(f"Failure: {e}")

    if(TopoDNN_RDSA_constits_clip_globalEnergy):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainingDataPath="Adversaries/TopoDNN/conserveConstits_spreadLimit_conserveGlobalEnergy/RDSA_train_data.npy",
                trainingTargetPath="Datasets/TopoDNN/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="RDSA_conserveConstits_spreadLimit_conserveGlobalEnergy"
            )
        except Exception as e:
            print(f"Failure: {e}")



    if(TopoDNN_FGSM_constits_clip_particleEnergy):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainingDataPath="Adversaries/TopoDNN/conserveConstits_spreadLimit_conserveParticleEnergy/FGSM_train_data.npy",
                trainingTargetPath="Datasets/TopoDNN/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="FGSM_conserveConstits_spreadLimit_conserveParticleEnergy"
            )
        except Exception as e:
            print(f"Failure: {e}")

    if(TopoDNN_PGD_constits_clip_particleEnergy):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainingDataPath="Adversaries/TopoDNN/conserveConstits_spreadLimit_conserveParticleEnergy/PGD_train_data.npy",
                trainingTargetPath="Datasets/TopoDNN/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="PGD_conserveConstits_spreadLimit_conserveParticleEnergy"
            )
        except Exception as e:
            print(f"Failure: {e}")

    if(TopoDNN_RDSA_constits_clip_particleEnergy):
        try:
            RetrainingDispatcher(
                baseModelPath="Models/TopoDNN/base_model.keras",
                retrainingDataPath="Adversaries/TopoDNN/conserveConstits_spreadLimit_conserveParticleEnergy/RDSA_train_data.npy",
                trainingTargetPath="Datasets/TopoDNN/train_target.npy",
                subdivisionCount=retraining_subdivisions,
                epochs=epochs,
                attackName="RDSA_conserveConstits_spreadLimit_conserveParticleEnergy"
            )
        except Exception as e:
            print(f"Failure: {e}")

    # endregion TopoDNN