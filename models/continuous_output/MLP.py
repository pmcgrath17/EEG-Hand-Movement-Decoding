from models.feature_calculation import feature_algorithms as processing
from models.feature_calculation import PCA_on_PSD as pca
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plotter
import sklearn

#########################
# Constants
path_to_data_file = "../../raw_data/"
label_file = "Data_Finger_Labels.txt"
data_file_template = "Data_EEG_"
sample_frequency = 1000
num_frequency_bins = 50
max_frequency = 30.0
num_channels = 16
#########################

if __name__ == '__main__':
    print("Running MLP")

    # Read in Data
    features = []
    labels = np.loadtxt(path_to_data_file + label_file)
    for index in range(len(labels)):
        feature_row = np.loadtxt(path_to_data_file + data_file_template + str(index) + ".txt")
        features += [np.transpose(feature_row, (1, 0))]
    features = np.array(features)


    # Construct frequency bins
    bin_width = max_frequency / num_frequency_bins
    freq_bins = np.zeros((num_frequency_bins, 2))
    for i in range(num_frequency_bins):
        freq_bins[i, 0] = i * bin_width
        freq_bins[i, 1] = (i + 1) * bin_width


    # Apply PSD to data
    processed_features = processing.average_PSD_algorithm(X=features, sample_freq=sample_frequency, bins=freq_bins)

    # Apply PCA to PSD data
    stat_matricies = pca.calc_covariance_matrices(processed_features)
    eig_vectors = pca.calc_eig_vects(stat_matricies)
    projected_features = pca.project_onto_pcs(processed_features, eig_vectors, num_frequency_bins)

    print(projected_features)

    # Shuffle the data
    features, labels = sklearn.utils.shuffle(projected_features, labels)

    # Train models (one for each finger)
    models = []
    histories = []
    #for i in range(5):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(num_channels, freq_bins.shape[0])),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear'),
        #tf.keras.layers.ReLU(max_value=1),
    ])
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=[tf.metrics.MeanSquaredError()]
    )

    history = model.fit(features, labels[:, 1], epochs=8, verbose=1, validation_split=0.4)
    histories += [history]

    # Evaluate model
    #######
    # There is no test-training split as of yet
    #######

    # Plot results

    train_acc = history.history['mean_squared_error']
    val_acc = history.history['val_mean_squared_error']

    fig, axes = plotter.subplots()
    axes.set_title('MLP Learning Plot With Linear Output')
    axes.plot(train_acc, label='training mse')
    axes.plot(val_acc, label='validation mse')
    axes.set_xlabel('Epochs')
    axes.set_ylabel('Mean Squared Error')
    axes.legend(loc='center right')
    plotter.show()