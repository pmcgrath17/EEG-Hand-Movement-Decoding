%% Load data and Constants
filename = 'C:\Users\mcgrathp\Documents\Undergrad_Research\EMG-Decoder-master\EMG-Decoder-master\python\saved_data\train_2020-09-15_21-21-27.json';
json = jsondecode(fileread(filename));

output_folder = 'C:\Users\mcgrathp\PycharmProjects\EEG-Hand-Movement-Decoding\raw_data\';

%% Loop through and calculate features at each timestep
n_features = 16;
N = length(json.raw_list);
z = zeros(N, n_features);

% flip buffer so that the oldest sample is at the top
buffer = flip(json.raw_list{1}, 1);
%z(1, :) = calc_feature(buffer);
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

save_labels(json.finger_data, output_folder);


% Grab subset of EEG samples to export
function small_buffer = buffer_for_export(full_buffer, len)
    num_samples = len - 30;
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
    shifted_finger_data = finger_data + 1;
    shifted_finger_data = shifted_finger_data / 2;
    output_file = sprintf('Data_Finger_Labels');
    output_file_name = strcat(output_folder, output_file);
    writematrix(shifted_finger_data, output_file_name, 'Delimiter', 'tab');
end