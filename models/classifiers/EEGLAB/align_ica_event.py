import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plotter
import sklearn.model_selection as model
from sklearn.svm import SVC

##################
feature_file = "/Users/patrickmcgrath/PycharmProjects/EEG-Hand-Movement-Decoding-Updated/models/classifiers/EEGLAB/pca_processed_features.csv"
label_file = "/Users/patrickmcgrath/PycharmProjects/EEG-Hand-Movement-Decoding-Updated/models/classifiers/EEGLAB/pca_processed_labels.csv"
sample_frequency = 1000
##################
# Utility function for Decoding

## Best parameters: C=10.0, gamma=.01
def grid_search_svm(X, y):
    C_range = np.logspace(-2, 3, 6)
    gamma_range = np.logspace(-3, -1, 3)
    param_grid = dict(gamma=gamma_range, C=C_range)
    #cv = model.StratifiedShuffleSplit(n_splits=10, test_size=0.15, random_state=4)
    cv = model.LeaveOneOut()
    grid = model.GridSearchCV(SVC(), param_grid=param_grid, cv=cv, verbose=2)
    grid.fit(X, y)
    return grid

def train_mlp(train_data, train_labels, hid_layer_nodes=None, epoch_cnt=None):
    if epoch_cnt is None:
        epoch_cnt = 100

    if hid_layer_nodes is None:
        hid_layer_nodes = 30

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=train_data.shape[1]),
        tf.keras.layers.Dense(hid_layer_nodes, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # output layer
        # Activation of each layer is linear (i.e. no activation:
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(train_data, train_labels, epochs=epoch_cnt, verbose=0)

    return model, history


##################

feature_data = np.loadtxt(feature_file, delimiter=',')
label_data = np.loadtxt(label_file, delimiter=',')

print(feature_data.shape)
print(label_data.shape)

feature_data = feature_data[:, :80] # Take first 40 principal components

training_size = int(.85 * len(feature_data))

training_data = feature_data[:training_size, :]
training_labels = label_data[:training_size]
testing_data = feature_data[training_size:, :]
testing_labels = label_data[training_size:]

print(training_data.shape)
print(testing_data.shape)
#grid_result = grid_search_svm(training_data, training_labels)
#grid_result = grid_search_svm(feature_data, label_data)

#print("The best parameters are %s with a score of %0.8f" % (grid_result.best_params_, grid_result.best_score_))
#print(grid_result.error_score)
#print(grid_result.score(testing_data, testing_labels))
fold_num = 1
kfold = model.KFold(n_splits=10, shuffle=True)
acc_per_fold = []
loss_per_fold = []
for train, test in kfold.split(feature_data, label_data):
    model, history = train_mlp(feature_data[train], label_data[train], epoch_cnt=200, hid_layer_nodes=120)
    scores = model.evaluate(feature_data[test], label_data[test])
    print(f'Score for fold {fold_num}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    fold_num = fold_num + 1

print(np.average(acc_per_fold))
print(np.average(loss_per_fold))