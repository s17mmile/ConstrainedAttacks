import gc
import os
import numpy as np
import keras
import sys
import tensorflow as tf
from keras import layers, models
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from Helpers.RDSA_Helpers import GetPDFsAndBinEdges, DistributionShuffleAttackPerVarsParallel
from Helpers.EvaluateModel import getFoolingRatio, getLossAndAccuracy, compute_difference_per_event, get_JSD
import multiprocessing
import tqdm
import random
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from itertools import repeat

if __name__ == "__main__":
    run = f'TestBaseModel'
    activation = 'relu'
    final_activation = 'sigmoid'
    loss = 'categorical_crossentropy'
    batchsize = 200
    epochs = 100
    lr = 0.000003

    # Step 1: Load the MNIST 784 dataset
    mnist = fetch_openml('mnist_784')

    # Prepare the data
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(int)

    # Normalize the features (Standardize)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Calculate the number of unique values for each feature (pixel)
    num_unique_values = [len(np.unique(X_scaled[:, i])) for i in range(X_scaled.shape[1])]

    indices_with_more_than_100_unique = [i for i, unique_count in enumerate(num_unique_values) if unique_count > 100]

    # One-hot encode the target variable
    y_encoded = to_categorical(y)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    input_shape = X_train.shape[1]

    print(type(input_shape))

    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=[input_shape]))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=lr), loss=loss,
                    metrics=['accuracy', 'AUC'])
    model.summary()


    # Step 3: Train the model

    runDirectory = f'./Models/{run}'
    if not os.path.exists(runDirectory):
        os.makedirs(runDirectory)

    # Checkpoint, saving only the best model
    saveModel = ModelCheckpoint(f'{runDirectory}/best_model.keras',
                                save_best_only=True,
                                monitor='val_loss',
                                mode='min')

    # Train the model if it doesn't exist yet, otherwise load the existing model

    if os.path.isfile(f'{runDirectory}/best_model.keras'):
        model = keras.models.load_model(f'{runDirectory}/best_model.keras')
        print("model already existed!")

    else:
        model.fit(X_train, y_train,
                    batch_size=batchsize,
                    epochs=epochs,
                    verbose=0,
                    callbacks=[saveModel],
                    validation_split=0.2,
                    shuffle=True)

    # Step 4: Evaluate the model on the test set
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Base Model Score: ", score)  # loss, accuracy, AUC


    # Save the model and evaluation metrics (Loss, Accuracy, Area Under ROC Curve)
    np.save(f'{runDirectory}/loss_base', score[0])
    np.save(f'{runDirectory}/acc_base', score[1])
    np.save(f'{runDirectory}/auc_base', score[2])

    model_path = f'{runDirectory}/best_model.keras'

    print("BEGINNING RDSA")

    # Step 5: Generate adversarial examples using the Random Distribution Shuffle Attack
    event_test = []
    print(X_test[0].shape)
    print(y_test[0].shape)
    for i in range(len(X_test)):
        event_test.append([X_test[i], y_test[i]])

    # Get the number of unique values for each feature (pixel)
    cont_vars = [i for i, unique_count in enumerate(num_unique_values) if unique_count > 100]
    categ_vars = np.delete(np.arange(0, input_shape), cont_vars)

    print(len(cont_vars))
    print(len(categ_vars))

    model = keras.models.load_model(model_path)

    # Get the probability distributions and bin edges for the continuous variables
    pdfs, bin_idxes, bin_edges = GetPDFsAndBinEdges(X_test, 100)


    perturbVars = 300   # Number of variables to perturb

    # Randomly select variables to perturb
    var_indices = [random.sample(cont_vars, perturbVars) for i in range(len(event_test))]

    num_workers = 4
    with multiprocessing.get_context("spawn").Pool(num_workers) as p:
        advs = p.starmap(DistributionShuffleAttackPerVarsParallel,
                            tqdm.tqdm(zip(event_test, repeat(model_path), repeat("CCE"), repeat(100),
                                        var_indices, repeat(pdfs), repeat(bin_idxes), repeat(bin_edges),
                                        repeat(num_unique_values)), total=len(event_test)), chunksize=1)

    advs = np.asarray(advs)

    # Get fooling ratio, loss, accuracy, and AUC for the adversarial examples
    fr_advs = getFoolingRatio(model, X_test, y_test, advs, 'CCE')
    loss_acc_advs = getLossAndAccuracy(model, advs, y_test)
    loss_advs = loss_acc_advs[0]
    acc_advs = loss_acc_advs[1]
    score_adv = model.evaluate(advs, y_test, verbose=0)

    # Get the number of rows
    num_rows = X_test.shape[0]
    # Compute the midpoint
    midpoint = num_rows // 2

    x_test_split, _ = X_test[:midpoint], X_test[midpoint:]
    _, advs_first_split = advs[:midpoint], advs[midpoint:]

    # Compute the JSD between the original and adversarial examples
    jsd = get_JSD(x_test_split, advs_first_split, input_shape)
    # Compute the mean change per event
    mean_change = compute_difference_per_event(X_test, advs)

    # Save the results
    np.save(f'{runDirectory}/fr_advs_ca_{perturbVars}', fr_advs)
    np.save(f'{runDirectory}/loss_advs_ca_{perturbVars}', loss_advs)
    np.save(f'{runDirectory}/acc_advs_ca_{perturbVars}', acc_advs)
    np.save(f'{runDirectory}/auc_advs_ca_{perturbVars}', score_adv[2])
    np.save(f'{runDirectory}/advs_ca_{perturbVars}', advs)
    np.save(f'{runDirectory}/jsd_advs_ca_{perturbVars}', jsd)
    np.save(f'{runDirectory}/meanChange_advs_ca_{perturbVars}', mean_change)

    # Print the results
    print("______________________________________________________________")
    print(f"FR RDSA #Vars={perturbVars}: {fr_advs}")
    print(f"AUC RDSA #Vars={perturbVars}: {score_adv[2]}")
    print(f"JSD RDSA #Vars={perturbVars}: {jsd}")
    print(f"Mean Change RDSA #Vars={perturbVars}: {mean_change}")
    print("______________________________________________________________")

    del advs
    gc.collect()

