%% MODIFY THIS TO DETERMINE HOW MANY EEG READINGS YOU WANT TO HAVE PER TXT
% FILE!!!
sample_size = 1000;
%% Load data and Constants
% These filepaths may need to be modified depending on where you have
% placed the github repository.
filename = 'C:\Users\mcgrathp\PycharmProjects\EEG-Hand-Movement-Decoding\raw_data\classification_data\training_02_11_2020.json';
json = jsondecode(fileread(filename));

output_folder = 'C:\Users\mcgrathp\PycharmProjects\EEG-Hand-Movement-Decoding\raw_data\text_data\';

%% Loop through and calculate features at each timestep
n_features = 16;
N = length(json.raw_list);
z = zeros(N, n_features);

% flip buffer so that the oldest sample is at the top
buffer = flip(json.raw_list{1}, 1);
save_data(buffer, output_folder, 0);
% %z(1, :) = calc_feature(buffer);
len = size(json.raw_list{1}, 1);
small_buffer = buffer_for_export(buffer, size(buffer, 1));
save_data(small_buffer, output_folder, 1);
for i=2:N
    % length of new raw data packet
    len = size(json.raw_list{i}, 1);
    
    % make room for the new samples
    buffer = circshift(buffer, -len);
    
    % insert the new samples
    buffer(end-len+1:end, :) = json.raw_list{i};
    
    % calculate features (For now, implementation is in Python...)
    %z(i, :) = calc_feature(buffer);
    
    small_buffer = buffer_for_export(buffer, size(buffer, 1));
    save_data(small_buffer, output_folder, i);
end

% % Save the continuous finger labels
% save_labels(json.finger_data, output_folder);

% Save the classified hand movement
save_classification_labels(json.finger_data, output_folder);

% Average out the dataset for smoothing
function averaged_buffer = average_dataset(buffer, sample_size)
    averaged_buffer = buffer;
    for i=sample_size + 1:length(buffer)
        averaged_buffer(i, :) = mean(buffer(i - sample_size:i, :));
    end
end

% Apply common average reference
% Take means of rows and subtract mean from the original matrix
function modified_matrix = common_average_reference(matrix)
    mean_rows = mean(matrix, 2);
    modified_matrix = matrix - mean_rows;
end

% Grab subset of EEG samples to export
function small_buffer = buffer_for_export(full_buffer, len)
    num_samples = len - 200;
    small_buffer = full_buffer(num_samples:len,:);
end

% Save EEG data to .txt file (each index corresponds to row number of
% finger labels)
function save_data(buffer, output_folder, index)
    output_file = sprintf('Data_EEG_%d', index - 1);
    output_file_name = strcat(output_folder, output_file);
    writematrix(buffer, output_file_name, 'Delimiter', 'tab');
end

% Save finger labels to .txt file
function save_labels(finger_data, output_folder)
    %shifted_finger_data = finger_data + 1;
    %shifted_finger_data = shifted_finger_data / 2;
    shifted_finger_data = finger_data;
    output_file = sprintf('Data_Finger_Labels');
    output_file_name = strcat(output_folder, output_file);
    writematrix(shifted_finger_data, output_file_name, 'Delimiter', 'tab');
end

% Save classifiaction finger labels to .txt file
% The labels are supplied as follows:
%   0: Open hand label
%   1: Closing hand label
%   2: Closed hand label
%   3: Openning hand label
% Note that the threshold for open and closed hand readings are different.
% This is because it is easy to get closed hand reading while the open hand
% readings are a bit more volatile and must have a lower threshold (based
% on magnitude) as a result.

function save_classification_labels(finger_data, output_folder)
    %shifted_finger_data = finger_data + 1;
    %shifted_finger_data = shifted_finger_data / 2;
    classification_finger_data = zeros(size(finger_data, 1), 1);
    
    if filename == 'C:\Users\mcgrathp\PycharmProjects\EEG-Hand-Movement-Decoding\raw_data\classification_data\training_02_11_2020.json'
        classification_finger_data(1) = 3;
    end
    
    for i=2:size(finger_data,1)
        non_thumb_measurements = finger_data(i, 2:5);
        average_hand_measurement = mean(non_thumb_measurements);
        % Closed hand has a label of 2
       if average_hand_measurement > .98
           classification_finger_data(i) = 2;
        % Open hand has a label of 0
       elseif average_hand_measurement < -.96
           classification_finger_data(i) = 0;
        % Closing hand has a label of 1
       elseif classification_finger_data(i-1) == 1 || classification_finger_data(i-1) == 0
           classification_finger_data(i) = 1;
        % Openning hand has a label of 3
       else
           classification_finger_data(i) = 3;
       end
    end

    output_file = sprintf('Classification_Hand_Labels');
    output_file_name = strcat(output_folder, output_file);
    writematrix(classification_finger_data, output_file_name, 'Delimiter', 'tab');
end