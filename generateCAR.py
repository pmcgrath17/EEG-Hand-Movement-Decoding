import numpy as np
import os

def load_file_into_array(file_number):

    file = 'EEG-Hand-Movement-Decoding-master/raw_data/Data_EEG_' + str(file_number) + '.txt'
    array = np.loadtxt(file)
    return array

def calculate_average(array, row):
    mean = np.mean(array[row])
    array[row] = array[row] - mean
    return array
def main():
    # determine how many raw_data files there are:
    path, directory, files = next(os.walk("EEG-Hand-Movement-Decoding-master/raw_data"))

    # subtract the Data_Finger_Labels.txt
    num_data_files = len(files) - 1

    # Iterates through number of Data_EEG.txt files, and then calculates the CAR.
    for i in range(0, num_data_files):
        channel_values = load_file_into_array(i)
        file = open("Data_EEG_" + str(i) + "CAR.txt", "a")
        # for each array, within the Data_EEG.txt file it calculates the average, and subtracts that value from all channels.
        for j in range(0, len(channel_values)):
            CAR_array = calculate_average(channel_values, j)
            print(CAR_array)
            np.savetxt(file, CAR_array)

main()
