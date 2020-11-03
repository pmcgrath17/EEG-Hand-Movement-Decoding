from models.feature_calculation import feature_algorithms as processing
from models.feature_calculation import PCA_on_PSD as pca
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plotter
import sklearn

#########################
# Constants
path_to_data_file = "../../raw_data/continuous_data"
label_file = "Data_Finger_Labels.txt"
data_file_template = "Data_EEG_"
sample_frequency = 1000
num_frequency_bins = 50
min_frequency = 5.0
max_frequency = 50.0
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
    bin_width = (max_frequency - min_frequency) / num_frequency_bins
    freq_bins = np.zeros((num_frequency_bins, 2))
    for i in range(num_frequency_bins):
        freq_bins[i, 0] = i * bin_width + min_frequency
        freq_bins[i, 1] = (i + 1) * bin_width + min_frequency


    # Apply PSD to data
    processed_features = processing.average_PSD_algorithm(X=features, sample_freq=sample_frequency, bins=freq_bins)

    # Apply PCA to PSD data
    stat_matricies = pca.calc_covariance_matrices(processed_features)
    eig_vectors = pca.calc_eig_vects(stat_matricies)
    projected_features = pca.project_onto_pcs(processed_features, eig_vectors, num_frequency_bins)


    ## Make Raw EEG Channel Graphs
    channel_info = np.array([])
    for channel_example in features:
        if channel_info.size == 0:
            channel_info = channel_example
        else:
            channel_info = np.concatenate((channel_info, channel_example), axis=1)
    print(channel_info.shape)

    for channel in range(num_channels):
        fig, axes = plotter.subplots()
        axes.plot(range(channel_info.shape[1]), channel_info[channel])
        axes.set_ylabel('EEG Signal (mV)')
        axes.set_xlabel('Sample Number')
        axes.set_title('Raw EEG Signals For Channel {}'.format(channel + 1))

    ## Make PSD EEG Channel Graphs
#    channel_info = np.array([])
#    for processed_feature in processed_features:
#        if channel_info.size == 0:
#            channel_info = np.expand_dims(processed_feature, axis=2)
#        else:
#            channel_info_temp = np.expand_dims(processed_feature, axis=2)
#            channel_info = np.concatenate((channel_info, channel_info_temp), axis=2)
#
#    for channel in range(num_channels):
#        fig, axis = plotter.subplots()
#        axis.set_title('PSD For Channel {}'.format(channel + 1))
#        axis.set_xlabel('Feature Number (Corresponding to Particular Label)')
#        axis.set_ylabel('Average PSD')
#        for frequency_bin in range(num_frequency_bins):
#            axis.plot(channel_info[channel, frequency_bin, :], label='Bin: {}'.format(freq_bins[frequency_bin]))
#        axis.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
    plotter.show()

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
                  metrics=[tf.metrics.RootMeanSquaredError()]
    )

    history = model.fit(features, labels[:, 1], epochs=8, verbose=1, validation_split=0.1)
    #histories += [history]

    # Evaluate model
    #######
    # There is no test-training split as of yet
    #######

    # Plot results

    train_acc = history.history['root_mean_squared_error']
    val_acc = history.history['val_root_mean_squared_error']

    #fig, axes = plotter.subplots()
    #axes.set_title('MLP Learning Plot With Linear Output')
    #axes.plot(train_acc, label='training rmse')
    #axes.plot(val_acc, label='validation rmse')
    #axes.set_xlabel('Epochs')
    #axes.set_ylabel('Root Mean Squared Error')
    #axes.legend(loc='center right')

    plotter.show()
