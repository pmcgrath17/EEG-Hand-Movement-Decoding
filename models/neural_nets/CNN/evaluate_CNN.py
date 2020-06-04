# This file contains a function to train/evaluate a convolutional neural network across subjects.


import numpy as np
import matplotlib.pyplot as plotter
from models.neural_nets import example_generation
from models.neural_nets.CNN import conv_neural_net


# NOT TO BE MODIFIED:
# path to data files:
path_to_data_file = "../../../MATLAB/biosig/Data_txt/"


# Function description: trains and evaluates a CNN for selected subjects (separate instance for each subject).
# Inputs:
#   subject_nums = array subject numbers to evaluate
#       size: (num_subjects, )
#   window_size_example = size of sliding window to create more examples, in seconds
#   stride_size_example = size of "stride" of sliding window to create more examples, in seconds
#   sample_freq = sampling frequency
#   num_conv_layers = number of convolutional layers
#       a max pooling layer is added after each convolutional layer
#   num_dense_layers = number of fully connected layers after convolutional layers
#   num_kernels = number of kernels for convolutional layers
#   kernel_size = size of convolutional kernels
#   pool_size = size of pooling filters for max pooling layers
#   num_hidden_nodes = number of nodes of fully connected layers (except last layer)
#   num_epochs = number of epochs to train for
#   batch_size = mini-batch size for training
#   window_size_PSD = size of sliding window to calculate PSD, in seconds
#   stride_size_PSD = size of "stride" of sliding window to calculate PSD, in seconds
#   max_freq = maximum frequency of PSD to consider
#   num_bins = number of frequency bins for average PSD calculation
#   PCA = parameter to select whether to apply PCA algorithm
#       if PCA == 1: PCA algorithm is applied
#       else: PCA algorithm is not applied
#   num_pcs = number of principal components (eigenvectors) to project onto
#       validity: num_pcs <= num_bins
#   matrix_type = parameter to select which type of statistical matrix to calculate:
#       if matrix type == 1: autocorrelation matrices are calculated
#       if matrix type == 2: autocovariance matrices are calculated
#       else: Pearson autocovariance matrices are calculated
#   small_param = a small number to ensure that log(0) does not occur for log-normalization
#   val_fract = fraction of data to use as validation set
#   test_fract = fraction of data to use as test set
#   standard = parameter to select whether to standardize features
#       if standard == True: features are standardized
#       else: features are not standardized
# Outputs:
#   avg_train_acc = average training accuracy across subjects
#   avg_val_acc = average validation accuracy across subjects
#   train_acc = dictionary of training accuracies for subjects
#   val_acc = dictionary of validation accuracies for subjects
def train_eval_CNN(subject_nums, window_size_example, stride_size_example, sample_freq, num_conv_layers,
                   num_dense_layers, num_kernels, kernel_size, pool_size, num_hidden_nodes, num_epochs, batch_size,
                   window_size_PSD, stride_size_PSD, max_freq, num_bins, PCA=0, num_pcs=None, matrix_type=2,
                   small_param=0.0001, val_fract=0.2, test_fract=0.15, standard=True):
    # number of subjects:
    num_subjects = subject_nums.shape[0]
    # dictionaries for training and validation accuracies for subjects:
    train_acc = {}
    val_acc = {}
    # average training and validation accuracies:
    avg_train_acc = 0
    avg_val_acc = 0

    # train a different CNN instance for each subject and record training/validation accuracies:
    for subject in subject_nums:
        print("\n\n\n--------------------SUBJECT {0}--------------------\n\n".format(subject))

        # get data and generate examples:
        X, Y = example_generation.generate_examples(subject, path_to_data_file, window_size_example,
                                                    stride_size_example, sample_freq)
        # display dimensions of raw data:
        print("Size of raw data set: ", end="")
        print(X.shape)

        # create ConvNet object:
        CNN = conv_neural_net.ConvNet(num_conv_layers, num_dense_layers, num_kernels, kernel_size, pool_size,
                                      num_hidden_nodes)
        # generate training and test features:
        X_train, Y_train, X_val, Y_val, X_test, Y_test = CNN.generate_features(X, Y, window_size_PSD, stride_size_PSD,
                                                                               sample_freq, max_freq, num_bins, PCA=PCA,
                                                                               num_pcs=num_pcs, matrix_type=matrix_type,
                                                                               small_param=small_param,
                                                                               val_fract=val_fract,
                                                                               test_fract=test_fract, standard=standard)
        print("Size of training set: ", end="")
        print(X_train.shape)
        print("Size of validation set: ", end="")
        print(X_val.shape)
        print("Size of test set: ", end="")
        print(X_test.shape)

        # build CNN model:
        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
        CNN.build_model(input_shape)
        # display model architecture:
        print("\n")
        CNN.model.summary()

        # train model:
        CNN.train_model(X_train, Y_train, X_val, Y_val, num_epochs, batch_size)
        # plot learning curve:
        CNN.plot_learn_curve(subject)

        # extract training and validation accuracies and record final values:
        training_acc = CNN.history.history['binary_accuracy']
        validation_acc = CNN.history.history['val_binary_accuracy']
        train_acc[subject] = training_acc[num_epochs-1]
        val_acc[subject] = validation_acc[num_epochs-1]
        # keep running sum of training and validation accuracies:
        avg_train_acc += train_acc[subject]
        avg_val_acc += val_acc[subject]

    # normalize averages:
    avg_train_acc = avg_train_acc / num_subjects
    avg_val_acc = avg_val_acc / num_subjects

    return avg_train_acc, avg_val_acc, train_acc, val_acc


# Function description: plots a bar graph of training and validation accuracies for selected subjects.
# Inputs:
#   train_acc = dictionary of training accuracies for subjects
#   val_acc = dictionary of validation accuracies for subjects
# Outputs: none
def plot_accuracies(train_acc, val_acc):
    # convert dictionaries to arrays:
    train_acc = np.array(list(train_acc.items()))
    val_acc = np.array(list(val_acc.items()))
    # extract subject numbers (keys):
    subject_nums = train_acc[:, 0]
    # extract accuracies (values):
    train_acc = train_acc[:, 1]
    val_acc = val_acc[:, 1]

    # number of subjects:
    num_subjects = subject_nums.shape[0]

    # create subject labels:
    subject_labels = []
    for subject in subject_nums:
        subject_labels.append('Subject ' + str(int(subject)))

    # create and format subplot:
    fig, ax = plotter.subplots()
    # plotter.subplots_adjust(hspace=1)
    # locations of labels:
    bin_loc = np.arange(num_subjects)
    # width of bars:
    width = 0.25

    # plot bar graph of training and validation accuracies as percentages:
    ax.bar(bin_loc - width/2, 100 * train_acc, width, label='training accuracy')
    ax.bar(bin_loc + width/2, 100 * val_acc, width, label='validation accuracy')
    ax.set_title('Training and Validation Accuracies Across Subjects')
    ax.set_ylabel('Accuracy (Percentage)')
    ax.set_xticks(bin_loc)
    ax.set_xticklabels(subject_labels)
    ax.legend(loc='upper right')
